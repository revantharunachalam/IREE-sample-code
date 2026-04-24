"""Compile a simple TensorFlow model to IREE and run inference.

This example shows the full pipeline:
  1. Define a TF SavedModel (or tf.Module).
  2. Import it to MLIR via iree-import-tf.
  3. Compile to .vmfb with iree-compiler.
  4. Run inference with iree-runtime.
"""

import numpy as np
import tensorflow as tf

from iree.compiler.tools import tf as iree_tf_tools
from iree.compiler import compile_str
import iree.runtime as ireert

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import save_vmfb, benchmark_module

# ---------------------------------------------------------------------------
# 1. Define a simple TensorFlow model
# ---------------------------------------------------------------------------

class LinearModel(tf.Module):
    """y = x @ W + b  (a single dense layer without activation)."""

    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        self.W = tf.Variable(
            tf.random.normal([in_features, out_features], seed=42), name="W"
        )
        self.b = tf.Variable(tf.zeros([out_features]), name="b")

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)]
    )
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, self.W) + self.b


def build_and_save_model(saved_model_dir: str) -> LinearModel:
    model = LinearModel(in_features=4, out_features=2)
    tf.saved_model.save(model, saved_model_dir)
    print(f"SavedModel written to: {saved_model_dir}")
    return model


# ---------------------------------------------------------------------------
# 2. Compile with IREE
# ---------------------------------------------------------------------------

def compile_savedmodel(saved_model_dir: str, target_backend: str = "llvm-cpu") -> bytes:
    """Import a TF SavedModel and compile it to IREE bytecode."""
    # iree-import-tf converts SavedModel -> MLIR (StableHLO / MHLO)
    mlir_module = iree_tf_tools.compile_saved_model(
        saved_model_dir,
        import_only=True,         # produce MLIR, don't compile yet
        exported_names=["forward"],
    )
    # compile MLIR -> .vmfb
    vmfb = compile_str(
        mlir_module,
        target_backends=[target_backend],
        input_type="mhlo",
    )
    return vmfb


# ---------------------------------------------------------------------------
# 3. Run inference
# ---------------------------------------------------------------------------

def run_inference(vmfb: bytes, driver: str = "local-task") -> None:
    config = ireert.Config(driver_name=driver)
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    module = ctx.modules.module

    batch = np.random.randn(3, 4).astype(np.float32)
    result = module.forward(batch)
    print(f"Input shape : {batch.shape}")
    print(f"Output shape: {np.array(result).shape}")
    print(f"Output      :\n{np.array(result)}")

    benchmark_module(module, "forward", [batch], iterations=200)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = str(pathlib.Path(tmpdir) / "linear_model")

        print("=== Step 1: Build TF model ===")
        build_and_save_model(saved_model_dir)

        print("\n=== Step 2: Compile to IREE vmfb ===")
        vmfb = compile_savedmodel(saved_model_dir, target_backend="llvm-cpu")
        save_vmfb(vmfb, "output/simple_tf_linear.vmfb")

        print("\n=== Step 3: Run inference ===")
        run_inference(vmfb)
