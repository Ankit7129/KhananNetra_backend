"""Utilities for making console output safe on Windows code pages."""

from __future__ import annotations

import builtins
import sys
from typing import Any

_PATCHED = False


def _sanitize(value: Any) -> Any:
    """Return the ASCII-safe version of *value*, dropping unsupported characters."""
    if isinstance(value, str):
        return value.encode("ascii", "ignore").decode("ascii")
    return value


class _SanitizedStream:
    """Proxy that strips non-ASCII characters before writing to the wrapped stream."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def write(self, data: Any) -> Any:
        if isinstance(data, str):
            data = _sanitize(data)
        return self._wrapped.write(data)

    def flush(self) -> None:
        self._wrapped.flush()

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        return getattr(self._wrapped, item)


def patch_console_outputs() -> None:
    """Ensure print/stdout/stderr never emit characters unsupported by the shell."""
    global _PATCHED
    if _PATCHED:
        return

    original_print = builtins.print

    def ascii_print(*args: Any, **kwargs: Any) -> None:
        sanitized_args = [_sanitize(arg) for arg in args]
        if "sep" in kwargs and isinstance(kwargs["sep"], str):
            kwargs["sep"] = _sanitize(kwargs["sep"])
        if "end" in kwargs and isinstance(kwargs["end"], str):
            kwargs["end"] = _sanitize(kwargs["end"])
        if "file" in kwargs and kwargs["file"] not in (None, sys.stdout, sys.stderr):
            original_print(*sanitized_args, **kwargs)
            return
        if "file" in kwargs:
            kwargs["file"] = kwargs["file"]  # keep explicit file
        original_print(*sanitized_args, **kwargs)

    builtins.print = ascii_print  # type: ignore
    sys.stdout = _SanitizedStream(sys.stdout)
    sys.stderr = _SanitizedStream(sys.stderr)
    _PATCHED = True
