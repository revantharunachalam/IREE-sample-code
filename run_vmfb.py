"""Run an IREE .vmfb from disk.

Examples:
  python run_vmfb.py output/matmul_add.vmfb --list
  python run_vmfb.py output/matmul_add.vmfb --function main --inputs a.npy b.npy bias.npy
  python run_vmfb.py output/matmul_add.vmfb --device vulkan --function main --inputs a.npy b.npy bias.npy

Notes:
  - Inputs are loaded from NumPy .npy files (one per function argument).
  - If you omit --function, it defaults to "main".
  - The vmfb must have been compiled for a target compatible with --device
    (e.g. llvm-cpu ↔ local-task, vulkan-spirv ↔ vulkan, cuda ↔ cuda).
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

import numpy as np
import iree.runtime as ireert


# IREE SystemContext always registers these modules by default (infrastructure
# modules, not user-compiled ones), so we skip them when auto-picking.
_BUILTIN_MODULE_NAMES = frozenset({"hal", "vmvx"})


def _extract_compiled_backend(err_msg: str) -> str | None:
    """Pull the executable target (e.g. 'llvm-cpu') out of an IREE INCOMPATIBLE error."""
    m = re.search(r'#hal\.executable\.target<"([^"]+)"', err_msg)
    return m.group(1) if m else None


def _load_vmfb_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _create_context(driver: str) -> ireert.SystemContext:
    config = ireert.Config(driver_name=driver)
    return ireert.SystemContext(config=config)


def _user_module_names(ctx: ireert.SystemContext) -> list[str]:
    # BoundModules is a dict subclass; iterate keys to get real module names.
    return [n for n in ctx.modules if n not in _BUILTIN_MODULE_NAMES]


def _safe_function_names(bound_module: Any) -> list[str]:
    """Best-effort enumeration of exported function names on a BoundModule."""
    vm = getattr(bound_module, "vm_module", None)
    if vm is None:
        return []
    for attr in ("function_names", "exported_names", "exports"):
        val = getattr(vm, attr, None)
        if val is None:
            continue
        if callable(val):
            try:
                val = val()
            except Exception:
                continue
        try:
            return list(val)
        except Exception:
            continue
    return []


def _pick_module(ctx: ireert.SystemContext, requested: str | None) -> Any:
    if requested:
        if requested not in ctx.modules:
            raise SystemExit(
                f"Module '{requested}' not found. Available: {list(ctx.modules)}"
            )
        return ctx.modules[requested]

    user_mods = _user_module_names(ctx)
    if not user_mods:
        raise SystemExit(
            f"No user module found in VMFB. All registered modules: {list(ctx.modules)}"
        )
    if len(user_mods) > 1:
        raise SystemExit(
            f"Multiple user modules found: {user_mods}. Pass --module <name>."
        )
    return ctx.modules[user_mods[0]]


def _pick_function(module: Any, requested: str | None) -> Any:
    # BoundModule uses __getattr__ for dynamic lookup; dir() won't list functions.
    name = requested or "main"
    try:
        return getattr(module, name)
    except AttributeError as e:
        fn_names = _safe_function_names(module)
        hint = f" Available: {fn_names}" if fn_names else ""
        raise SystemExit(f"Function '{name}' not found on module.{hint}") from e


def _to_host(x: Any) -> np.ndarray:
    # IREE device arrays expose .to_host(), which avoids the NumPy 2.x
    # "__array__ doesn't accept copy keyword" DeprecationWarning.
    if hasattr(x, "to_host"):
        return x.to_host()
    return np.asarray(x)


def _list_exports(ctx: ireert.SystemContext) -> None:
    all_names = list(ctx.modules)
    print("Available modules:")
    for mn in all_names:
        marker = " (builtin)" if mn in _BUILTIN_MODULE_NAMES else ""
        print(f"  - {mn}{marker}")
    for mn in all_names:
        if mn in _BUILTIN_MODULE_NAMES:
            continue
        fns = _safe_function_names(ctx.modules[mn])
        if fns:
            print(f"\nExports in module '{mn}':")
            for fn in fns:
                print(f"  - {fn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an IREE .vmfb from disk.")
    parser.add_argument(
        "vmfb",
        help="Path to .vmfb (e.g. output/matmul_add.vmfb)",
    )
    parser.add_argument(
        "--device",
        default="local-task",
        help="IREE runtime driver: local-task (CPU), vulkan, cuda, rocm, etc.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List modules and exported functions, then exit.",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Module name to select (required only if vmfb contains multiple user modules).",
    )
    parser.add_argument(
        "--function",
        default=None,
        help="Function name to invoke (defaults to 'main').",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Input tensors as .npy files (one per argument).",
    )
    args = parser.parse_args()

    vmfb_path = os.path.normpath(args.vmfb)
    if not os.path.exists(vmfb_path):
        raise SystemExit(f"VMFB not found: {vmfb_path}")

    vmfb_bytes = _load_vmfb_bytes(vmfb_path)

    ctx = _create_context(args.device)
    try:
        ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes))
    except RuntimeError as e:
        err_msg = str(e)
        if "INCOMPATIBLE" in err_msg:
            compiled_for = _extract_compiled_backend(err_msg) or "an unknown backend"
            raise SystemExit(
                f"VMFB '{vmfb_path}' was compiled for {compiled_for}, which is not "
                f"compatible with --device '{args.device}'.\n"
                f"Either re-run with a matching device (e.g. --device local-task for "
                f"llvm-cpu), or recompile the MLIR with a matching target_backends "
                f"(e.g. 'vulkan-spirv' for --device vulkan, 'cuda' for --device cuda)."
            ) from e
        raise

    if args.list:
        _list_exports(ctx)
        return

    module = _pick_module(ctx, args.module)
    fn = _pick_function(module, args.function)

    missing = [p for p in args.inputs if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Input .npy files not found: "
            + ", ".join(missing)
            + "\nCreate them with NumPy, e.g.:\n"
            + '  python -c "import numpy as np; '
            + "np.save('a.npy', np.random.randn(4,8).astype(np.float32)); "
            + "np.save('b.npy', np.random.randn(8,4).astype(np.float32)); "
            + "np.save('bias.npy', np.random.randn(4).astype(np.float32))\""
        )

    np_inputs = [np.load(p) for p in args.inputs]

    result = fn(*np_inputs)
    if isinstance(result, tuple):
        result = tuple(_to_host(x) for x in result)
    else:
        result = _to_host(result)

    print("Result:")
    print(result)


if __name__ == "__main__":
    main()
