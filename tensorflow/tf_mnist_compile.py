"""Compile a Keras MNIST classifier to IREE.

Pipeline:
  1. Train (or load) a small Conv2D + Dense Keras model.
  2. Export as SavedModel.
  3. Compile to .vmfb via iree-import-tf + iree-compiler.
  4. Validate outputs against native Keras predictions.
"""

import numpy as np
import tensorflow as tf

from iree.compiler.tools import tf as iree_tf_tools
from iree.compiler import compile_str
import iree.runtime as ireert

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import save_vmfb

# ---------------------------------------------------------------------------
# 1. Build a small MNIST-style CNN
# ---------------------------------------------------------------------------

def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1), name="image"),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax", name="logits"),
        ],
        name="mnist_cnn",
    )
    return model


def train_model(model: tf.keras.Model, epochs: int = 1) -> tf.keras.Model:
    """Train on MNIST for a single epoch (demo only)."""
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype("float32") / 255.0

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train[:5000], y_train[:5000], batch_size=64, epochs=epochs, verbose=1)
    return model


# ---------------------------------------------------------------------------
# 2. Wrap as tf.Module so IREE can export a named function
# ---------------------------------------------------------------------------

class MNISTModule(tf.Module):
    def __init__(self, keras_model: tf.keras.Model):
        super().__init__()
        self.model = keras_model

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32)]
    )
    def predict(self, images: tf.Tensor) -> tf.Tensor:
        return self.model(images, training=False)


# ---------------------------------------------------------------------------
# 3. Compile to IREE
# ---------------------------------------------------------------------------

def compile_mnist(saved_model_dir: str, target_backend: str = "llvm-cpu") -> bytes:
    mlir_module = iree_tf_tools.compile_saved_model(
        saved_model_dir,
        import_only=True,
        exported_names=["predict"],
    )
    return compile_str(
        mlir_module,
        target_backends=[target_backend],
        input_type="mhlo",
    )


# ---------------------------------------------------------------------------
# 4. Validate against Keras reference
# ---------------------------------------------------------------------------

def validate(vmfb: bytes, keras_model: tf.keras.Model, n_samples: int = 8) -> None:
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = x_test[:n_samples][..., np.newaxis].astype("float32") / 255.0

    # Keras reference
    keras_probs = keras_model.predict(x, verbose=0)
    keras_preds = np.argmax(keras_probs, axis=-1)

    # IREE inference
    config = ireert.Config(driver_name="local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    iree_probs = np.array(ctx.modules.module.predict(x))
    iree_preds = np.argmax(iree_probs, axis=-1)

    print(f"Ground truth  : {y_test[:n_samples].tolist()}")
    print(f"Keras preds   : {keras_preds.tolist()}")
    print(f"IREE preds    : {iree_preds.tolist()}")
    match = np.allclose(keras_probs, iree_probs, atol=1e-5)
    print(f"Outputs match : {match}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = str(pathlib.Path(tmpdir) / "mnist_model")

        print("=== Step 1: Build & train model ===")
        keras_model = build_model()
        keras_model = train_model(keras_model, epochs=1)

        print("\n=== Step 2: Export SavedModel ===")
        module = MNISTModule(keras_model)
        tf.saved_model.save(module, saved_model_dir)

        print("\n=== Step 3: Compile to IREE ===")
        vmfb = compile_mnist(saved_model_dir, target_backend="llvm-cpu")
        save_vmfb(vmfb, "output/mnist_cnn.vmfb")

        print("\n=== Step 4: Validate ===")
        validate(vmfb, keras_model)
