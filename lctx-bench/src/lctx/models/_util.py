"""Small shared helpers for adapters."""

from __future__ import annotations

import importlib
from typing import Any


def load_object(path: str) -> Any:
    """Import an object from a ``package.module:attribute`` path.

    Used wherever the config names user code — custom attention patches,
    custom model loaders — so that plugging in an implementation never requires
    editing lctx-bench itself.
    """
    if ":" not in path:
        raise ValueError(
            f"{path!r} must be of the form 'package.module:attribute' "
            "(for example 'myproj.attn:patch_model')"
        )
    module_path, _, attr = path.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"could not import module {module_path!r} from {path!r}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"module {module_path!r} has no attribute {attr!r}") from exc
