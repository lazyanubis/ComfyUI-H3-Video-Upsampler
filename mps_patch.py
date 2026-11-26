"""MPS (Apple Silicon) compatibility shims for H3 Video Upsampler.

Imported from ``__init__.py`` right after the node modules load. Nothing here
edits the upstream files, so re-vendoring them from the author's build tool
will not conflict with these changes.

What it fixes on Apple Silicon:

1. ``TOKEN_BUDGET`` — upstream picks 70000 tokens per tile forward, a number
   tuned for 16-24 GB NVIDIA cards. We lower it (overridable through
   ``H3UPS_TOKEN_BUDGET``) so tiles stay inside a 96 GB wired limit without
   spending an hour in over-tiling.
2. Out-of-memory handling — upstream only retries/falls back when
   ``device.type == "cuda"``; on MPS the raw ``RuntimeError`` escapes with no
   guidance. We keep the raw error as the cause and re-raise it with the
   knob to turn.
3. ``face_pass`` — upstream downloads YuNet from GitHub on first use. On this
   machine the network is unavailable, so the download only adds a timeout.
   We use a local copy when present and skip the feature otherwise.
"""

from __future__ import annotations

import os

TAG = "H3UpsamplerMPS"

#: Tile budget used on MPS when the widget leaves ``tile_tokens`` at 0.
DEFAULT_TOKEN_BUDGET = 24576


def _is_apple_silicon() -> bool:
    """Whether this process runs on Apple Silicon with a usable MPS device."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def _budget() -> int:
    """Token budget for MPS, from ``H3UPS_TOKEN_BUDGET`` when it is set."""
    raw = os.environ.get("H3UPS_TOKEN_BUDGET", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_BUDGET
    return value if value > 0 else DEFAULT_TOKEN_BUDGET


def _patch_budget(tiled) -> str | None:
    """Replace too-large tile budget with an MPS-sized one."""
    original = getattr(tiled, "TOKEN_BUDGET", None)
    if not isinstance(original, int):
        return None
    budget = _budget()
    tiled.TOKEN_BUDGET = budget
    return f"tile budget {original} -> {budget}"


def _patch_oom(tiled) -> str | None:
    """Give MPS out-of-memory failures an actionable message."""
    original = getattr(tiled, "upscale_chunk", None)
    if not callable(original):
        return None

    def upscale_chunk(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" not in message and "alloc" not in message:
                raise
            raise RuntimeError(
                "H3 upsampler ran out of memory on the MPS device. Lower "
                "tile_tokens (try 16384, then 12288), keep chunking and tiling "
                "on, and run a shorter clip first."
            ) from error

    upscale_chunk.__name__ = getattr(original, "__name__", "upscale_chunk")
    upscale_chunk.__doc__ = getattr(original, "__doc__", None)
    tiled.upscale_chunk = upscale_chunk
    return "MPS out-of-memory message"


def _patch_face_pass(face) -> str | None:
    """Use a local YuNet file only; never reach the network."""
    if face is None or not hasattr(face, "ensure_model"):
        return None

    def ensure_model():
        path = face._model_path()
        try:
            if os.path.isfile(path) and face._sha256(path) == face.YUNET_SHA256:
                return path
        except Exception:  # noqa: BLE001
            pass
        print(
            f"[{TAG}] small-face re-draw off: no local {face.YUNET_FILE}; "
            "downloads are disabled on this machine",
            flush=True,
        )
        return None

    face.ensure_model = ensure_model
    return "face detector stays offline"


def apply(tiled, face=None) -> None:
    """Install every MPS shim that applies to the loaded modules."""
    if not _is_apple_silicon():
        return
    notes = [note for note in (_patch_budget(tiled), _patch_oom(tiled),
                               _patch_face_pass(face)) if note]
    if notes:
        print(f"[{TAG}] {', '.join(notes)}", flush=True)
