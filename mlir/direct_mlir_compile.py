"""Compile hand-written MLIR directly to IREE.

Useful for understanding the compilation pipeline without a framework frontend.
Demonstrates:
  - Authoring MLIR with StableHLO ops.
  - Compiling to .vmfb.
  - Running with iree-runtime.
"""

import numpy as np
from iree.compiler import compile_str
import iree.runtime as ireert

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import save_vmfb, benchmark_module

# ---------------------------------------------------------------------------
# 1. Hand-written MLIR modules
# ---------------------------------------------------------------------------

# Matrix multiply + add: C = A @ B + bias
MATMUL_MLIR = r"""
module @matmul_add {
  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>, %bias: tensor<4xf32>)
      -> tensor<4x4xf32> {
    %mm = stablehlo.dot %a, %b : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %bias_bcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<4xf32>) -> tensor<4x4xf32>
    %result = stablehlo.add %mm, %bias_bcast : tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
"""

# Element-wise ReLU: max(x, 0)
RELU_MLIR = r"""
module @relu {
  func.func @relu(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<4x4xf32>
    %result = stablehlo.maximum %x, %zero : tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
"""

# Softmax over last dimension
SOFTMAX_MLIR = r"""
module @softmax {
  func.func @softmax(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // exp(x)
    %exp_x = stablehlo.exponential %x : tensor<2x4xf32>
    // sum over last dim
    %init = stablehlo.constant dense<0.0> : tensor<2xf32>
    %sum = stablehlo.reduce(%exp_x init: %init) applies stablehlo.add
        across dimensions = [1] : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2xf32>
    // broadcast sum back and divide
    %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0]
        : (tensor<2xf32>) -> tensor<2x4xf32>
    %result = stablehlo.divide %exp_x, %sum_bcast : tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}
"""


# ---------------------------------------------------------------------------
# 2. Compile helper
# ---------------------------------------------------------------------------

def compile_module(mlir_text: str, target_backend: str = "llvm-cpu") -> bytes:
    return compile_str(
        mlir_text,
        target_backends=[target_backend],
        input_type="stablehlo",
    )


# ---------------------------------------------------------------------------
# 3. Demo runners
# ---------------------------------------------------------------------------

def demo_matmul_add() -> None:
    print("--- matmul_add ---")
    vmfb = compile_module(MATMUL_MLIR)
    save_vmfb(vmfb, "output/matmul_add.vmfb")

    config = ireert.Config(driver_name="local-task")
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    m = ctx.modules.module

    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8, 4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)

    iree_result = np.array(m.main(a, b, bias))
    np_result = a @ b + bias

    print(f"Max diff from NumPy: {np.max(np.abs(iree_result - np_result)):.6f}")
    benchmark_module(m, "main", [a, b, bias], iterations=500)


def demo_relu() -> None:
    print("\n--- relu ---")
    vmfb = compile_module(RELU_MLIR)
    save_vmfb(vmfb, "output/relu.vmfb")

    config = ireert.Config(driver_name="local-task")
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    m = ctx.modules.module

    x = np.array([[-1, 2, -3, 4], [0, -1, 2, -3]], dtype=np.float32).reshape(4, 4)
    # pad to 4x4 for the static module
    x = np.random.randn(4, 4).astype(np.float32)
    result = np.array(m.relu(x))
    print(f"Input min: {x.min():.3f}, Output min (should be >= 0): {result.min():.3f}")


def demo_softmax() -> None:
    print("\n--- softmax ---")
    vmfb = compile_module(SOFTMAX_MLIR)
    save_vmfb(vmfb, "output/softmax.vmfb")

    config = ireert.Config(driver_name="local-task")
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    m = ctx.modules.module

    x = np.array([[1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    result = np.array(m.softmax(x))
    print(f"Softmax output (rows sum to 1):\n{result}")
    print(f"Row sums: {result.sum(axis=1)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_matmul_add()
    demo_relu()
    demo_softmax()
