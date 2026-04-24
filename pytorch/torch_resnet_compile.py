"""Compile torchvision ResNet-18 to IREE via torch-mlir.

Pipeline:
  1. Load pretrained ResNet-18 from torchvision.
  2. Export to MLIR using torch_mlir (FX/Dynamo path for better coverage).
  3. Compile to .vmfb.
  4. Compare IREE logits against native PyTorch.
"""

import numpy as np
import torch
import torchvision.models as models

import torch_mlir
from iree.compiler import compile_str
import iree.runtime as ireert

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import save_vmfb, benchmark_module

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------

def load_resnet18() -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 2. Export to MLIR (Dynamo / FX-based path)
# ---------------------------------------------------------------------------

def export_resnet_mlir(model: torch.nn.Module, batch_size: int = 1) -> str:
    """Use torch.export + torch_mlir to produce Linalg-on-tensors MLIR."""
    example_input = torch.randn(batch_size, 3, 224, 224)

    # torch.export produces a clean FX graph (PyTorch 2.x)
    exported = torch.export.export(model, (example_input,))

    mlir_module = torch_mlir.extras.fx_importer.export_and_import(
        exported,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    )
    return mlir_module.operation.get_asm(large_elements_limit=None)


# ---------------------------------------------------------------------------
# 3. Compile to IREE
# ---------------------------------------------------------------------------

def compile_to_iree(mlir_asm: str, target_backend: str = "llvm-cpu") -> bytes:
    return compile_str(
        mlir_asm,
        target_backends=[target_backend],
        input_type="linalg",
    )


# ---------------------------------------------------------------------------
# 4. Validate
# ---------------------------------------------------------------------------

def validate(vmfb: bytes, model: torch.nn.Module, batch_size: int = 1) -> None:
    config = ireert.Config(driver_name="local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    iree_module = ctx.modules.module

    x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    iree_logits = np.array(iree_module.main(x))  # torch.export names entry "main"

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x)).numpy()

    iree_top5 = np.argsort(iree_logits[0])[-5:][::-1]
    torch_top5 = np.argsort(torch_logits[0])[-5:][::-1]

    print(f"IREE  top-5 class indices : {iree_top5.tolist()}")
    print(f"Torch top-5 class indices : {torch_top5.tolist()}")
    print(f"Max logit diff            : {np.max(np.abs(iree_logits - torch_logits)):.6f}")

    benchmark_module(iree_module, "main", [x], iterations=10)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Step 1: Load ResNet-18 ===")
    model = load_resnet18()

    print("\n=== Step 2: Export to MLIR ===")
    mlir_asm = export_resnet_mlir(model, batch_size=1)

    print("\n=== Step 3: Compile to IREE vmfb ===")
    vmfb = compile_to_iree(mlir_asm, target_backend="llvm-cpu")
    save_vmfb(vmfb, "output/resnet18.vmfb")

    print("\n=== Step 4: Validate ===")
    validate(vmfb, model)
