"""Shared utilities: config loading, seeding, devices, logging, IO."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

LOGGER_NAME = "aes"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def progress(iterable, **kwargs):
    """Wrap an iterable in a ``tqdm`` bar with elapsed/remaining-time estimates.

    Falls back to the plain iterable when tqdm is unavailable or when
    ``TQDM_DISABLE=1`` is set (useful for non-interactive logs).

    ``mininterval`` defaults to 10 s so that notebook-style log capture
    (Kaggle/Colab cells) does not print one line per batch; on a terminal the
    bar updates just as often as before, only less frequently.
    """
    disabled = os.environ.get("TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
    if disabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        return iterable
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("mininterval", 10.0)
    return tqdm(iterable, **kwargs)


# --------------------------------------------------------------------------- #
# Seeding / determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if available) PyTorch/CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Device / environment
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto"):
    """Return a ``torch.device``. ``prefer`` in {auto, cuda, mps, cpu}."""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if prefer == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_environment() -> Dict[str, Any]:
    import platform

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["cuda_total_memory_gb"] = round(props.total_memory / 1e9, 2)
            info["cuda_capability"] = f"{props.major}.{props.minor}"
    except ImportError:
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    return info


def supports_amp(device) -> bool:
    """True when mixed precision is safe/beneficial (CUDA only here)."""
    import torch

    return device.type == "cuda" and torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, optionally inheriting from another config.

    A config may declare ``inherits: _base.yaml``; the parent is loaded first
    (recursively) and the child's keys are deep-merged on top.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg)}")

    parent = cfg.pop("inherits", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        base = load_config(parent_path)
        cfg = merge_overrides(base, cfg)
    return cfg


def deep_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``overrides`` into ``cfg`` (overrides win)."""
    out = json.loads(json.dumps(cfg))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_overrides(out[key], value)
        else:
            out[key] = value
    return out


def parse_key_value_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse ``a.b=1 c.d=hello`` into nested dict with YAML scalar parsing."""
    import yaml

    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    from .metrics import to_plain_floats

    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_plain_floats(obj), fh, indent=2, allow_nan=True)
        fh.write("\n")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def resolve_path(path: str | Path, base: Optional[str | Path] = None) -> Path:
    """Resolve a possibly-relative path against ``base`` or the repo root."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    root = Path(base) if base is not None else PROJECT_ROOT
    return (Path(root) / path).resolve()


def project_root() -> Path:
    return PROJECT_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
