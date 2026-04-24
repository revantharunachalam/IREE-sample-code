# IREE-sample-code

Sample project demonstrating how to compile and run TensorFlow and PyTorch models
using [IREE](https://iree.dev) (Intermediate Representation Execution Environment).

## Project layout

```
├── common/
│   └── iree_utils.py          # Shared helpers: compile, load, benchmark
├── tensorflow/
│   ├── simple_tf_model.py     # Linear layer tf.Module → .vmfb
│   └── tf_mnist_compile.py    # Keras MNIST CNN → .vmfb + validation
├── pytorch/
│   ├── simple_torch_model.py  # nn.Linear → torch-mlir → .vmfb
│   └── torch_resnet_compile.py# ResNet-18 (torchvision) → torch-mlir → .vmfb
├── mlir/
│   └── direct_mlir_compile.py # Hand-written StableHLO → .vmfb (no framework)
├── output/                    # Generated .vmfb artefacts (git-ignored)
└── requirements.txt
```

## Installation

```bash
# Create a virtual environment (Python 3.10 recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU targets**: replace `iree-compiler` with a GPU-enabled build and change
> `target_backend="llvm-cpu"` → `"vulkan-spirv"` or `"cuda"` in each script.

## Quick start

### Direct MLIR (no framework required)

```bash
python mlir/direct_mlir_compile.py
```

Compiles hand-written StableHLO MLIR for matmul+add, ReLU, and softmax.

### TensorFlow

```bash
# Simple linear model
python tensorflow/simple_tf_model.py

# MNIST CNN (downloads dataset on first run)
python tensorflow/tf_mnist_compile.py
```

### PyTorch

```bash
# Simple linear layer
python pytorch/simple_torch_model.py

# ResNet-18 (downloads pretrained weights on first run)
python pytorch/torch_resnet_compile.py
```

## Compilation pipeline

```
TensorFlow SavedModel
    └─► iree-import-tf (iree.compiler.tools.tf)
            └─► MHLO/StableHLO MLIR
                    └─► iree-compiler  ──► .vmfb
                                               └─► iree-runtime

PyTorch nn.Module
    └─► torch.export / TorchScript
            └─► torch-mlir (Linalg-on-tensors)
                    └─► iree-compiler  ──► .vmfb
                                               └─► iree-runtime

Hand-written MLIR (StableHLO)
    └─► iree-compiler  ──► .vmfb
                               └─► iree-runtime
```

## Target backends

| Backend         | `target_backend` value | Hardware         |
|-----------------|------------------------|------------------|
| CPU (default)   | `llvm-cpu`             | x86 / ARM        |
| Vulkan (GPU)    | `vulkan-spirv`         | Vulkan-capable   |
| CUDA            | `cuda`                 | NVIDIA GPU       |
| Metal           | `metal-spirv`          | Apple GPU        |
| WebGPU          | `webgpu-spirv`         | Browser / WASM   |

## Key IREE concepts

- **`iree.compiler`** — Python bindings for the IREE compiler. Accepts MLIR text
  or bytecode and emits a `.vmfb` (VM FlatBuffer) artifact.
- **`iree.runtime`** — Python bindings for the IREE runtime. Loads `.vmfb` files,
  manages HAL devices, and exposes compiled functions as Python callables.
- **HAL driver** — hardware abstraction layer. `local-task` runs on CPU via a
  thread-pool; other drivers map to Vulkan / CUDA / Metal.
- **`torch-mlir`** — converts PyTorch `nn.Module` (via TorchScript or
  `torch.export`) to MLIR dialects that IREE can consume.
- **`iree-tools-tf`** — wraps `iree-import-tf` to convert TF SavedModels to MLIR.

## References

- IREE documentation: https://iree.dev
- torch-mlir: https://github.com/llvm/torch-mlir
- IREE GitHub: https://github.com/iree-org/iree
