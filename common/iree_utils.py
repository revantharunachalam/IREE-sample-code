"""Shared utilities for IREE compilation and execution."""

import os
import numpy as np
import iree.runtime as ireert
from iree.compiler import compile_str


def compile_mlir_module(mlir_source: str, target_backend: str = "llvm-cpu") -> bytes:
    """Compile an MLIR module string to IREE bytecode.

    Args:
        mlir_source: MLIR module as a string.
        target_backend: IREE target backend (llvm-cpu, vulkan-spirv, cuda).

    Returns:
        Compiled IREE bytecode as bytes.
    """
    return compile_str(
        mlir_source,
        target_backends=[target_backend],
        input_type="stablehlo",
    )


def load_vmfb(vmfb_bytes: bytes, driver: str = "local-task") -> ireert.VmModule:
    """Load compiled IREE bytecode into a runtime module.

    Args:
        vmfb_bytes: Compiled IREE bytecode (.vmfb).
        driver: IREE HAL driver (local-task, vulkan, cuda).

    Returns:
        IREE VmModule ready for invocation.
    """
    config = ireert.Config(driver_name=driver)
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes)
    ctx.add_vm_module(vm_module)
    return ctx.modules.module


def save_vmfb(vmfb_bytes: bytes, path: str) -> None:
    """Save compiled bytecode to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(vmfb_bytes)
    print(f"Saved compiled module to: {path}")


def load_vmfb_from_file(path: str, driver: str = "local-task") -> ireert.VmModule:
    """Load a .vmfb file from disk into a runtime module."""
    with open(path, "rb") as f:
        vmfb_bytes = f.read()
    return load_vmfb(vmfb_bytes, driver=driver)


def benchmark_module(module, fn_name: str, inputs: list, iterations: int = 100) -> float:
    """Run a simple latency benchmark for a compiled function.

    Returns average latency in milliseconds.
    """
    import time

    fn = getattr(module, fn_name)
    # warm-up
    for _ in range(5):
        fn(*inputs)

    start = time.perf_counter()
    for _ in range(iterations):
        fn(*inputs)
    elapsed_ms = (time.perf_counter() - start) / iterations * 1000
    print(f"{fn_name}: avg latency = {elapsed_ms:.3f} ms over {iterations} iterations")
    return elapsed_ms
