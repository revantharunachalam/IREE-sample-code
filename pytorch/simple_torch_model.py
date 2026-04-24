"""Compile a simple PyTorch model to IREE via torch-mlir.

Pipeline:
  1. Define a torch.nn.Module.
  2. Export to MLIR using torch_mlir.compile (TorchScript path).
  3. Compile the MLIR to .vmfb with iree-compiler.
  4. Run inference with iree-runtime.
"""

import numpy as np
import torch
import torch.nn as nn

import torch_mlir
from iree.compiler import compile_str
import iree.runtime as ireert

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import save_vmfb, benchmark_module


# ---------------------------------------------------------------------------
# 1. Define a simple PyTorch model
# ---------------------------------------------------------------------------

class LinearModel(nn.Module):
    """Single linear layer: y = xW^T + b."""

    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# 2. Compile to MLIR and then to IREE
# ---------------------------------------------------------------------------

def torch_model_to_mlir(model: nn.Module, example_inputs: tuple) -> str:
    """Convert a PyTorch module to MLIR via torch-mlir (TorchScript backend)."""
    model.eval()
    mlir_module = torch_mlir.compile(
        model,
        example_inputs,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,   # TorchScript path; set True for non-scriptable models
    )
    return mlir_module.operation.get_asm(large_elements_limit=None)


def compile_to_iree(mlir_asm: str, target_backend: str = "llvm-cpu") -> bytes:
    return compile_str(
        mlir_asm,
        target_backends=[target_backend],
        input_type="linalg",
    )


# ---------------------------------------------------------------------------
# 3. Run inference
# ---------------------------------------------------------------------------

def run_inference(vmfb: bytes, model: nn.Module, driver: str = "local-task") -> None:
    config = ireert.Config(driver_name=driver)
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    iree_module = ctx.modules.module

    batch = np.random.randn(3, 4).astype(np.float32)

    # IREE result
    iree_out = np.array(iree_module.forward(batch))

    # PyTorch reference
    with torch.no_grad():
        torch_out = model(torch.from_numpy(batch)).numpy()

    print(f"Input  shape : {batch.shape}")
    print(f"Output shape : {iree_out.shape}")
    print(f"IREE   output:\n{iree_out}")
    print(f"Torch  output:\n{torch_out}")
    print(f"Max diff     : {np.max(np.abs(iree_out - torch_out)):.6f}")

    benchmark_module(iree_module, "forward", [batch], iterations=200)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Step 1: Build PyTorch model ===")
    model = LinearModel(in_features=4, out_features=2)

    print("\n=== Step 2: Export to MLIR ===")
    example_inputs = (torch.randn(1, 4),)
    mlir_asm = torch_model_to_mlir(model, example_inputs)

    print("\n=== Step 3: Compile to IREE vmfb ===")
    vmfb = compile_to_iree(mlir_asm, target_backend="llvm-cpu")
    save_vmfb(vmfb, "output/simple_torch_linear.vmfb")

    print("\n=== Step 4: Run inference ===")
    run_inference(vmfb, model)
