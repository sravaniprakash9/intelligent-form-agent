"""HuggingFace connectivity and model download helpers."""

from __future__ import annotations

import socket
from functools import lru_cache

SURYA_MODELS = (
    "vikp/surya_det2",
    "vikp/surya_rec",
)


@lru_cache(maxsize=1)
def is_huggingface_reachable(timeout: float = 3.0) -> bool:
    """Quick DNS/connectivity check for huggingface.co."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
        return True
    except OSError:
        return False


def surya_models_cached() -> bool:
    """Return True if Surya model files appear in the local HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        for repo in SURYA_MODELS:
            path = try_to_load_from_cache(repo, "config.json")
            if path is None:
                return False
        return True
    except Exception:
        return False


def can_use_surya() -> bool:
    """Surya can run if models are cached locally or HuggingFace is reachable."""
    return surya_models_cached() or is_huggingface_reachable()


def download_surya_models() -> None:
    """Pre-download Surya OCR models from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    if not is_huggingface_reachable():
        raise RuntimeError(
            "Cannot reach huggingface.co. Check internet/DNS/VPN, then retry.\n"
            "Try: nslookup huggingface.co  OR  curl -I https://huggingface.co"
        )

    for repo in SURYA_MODELS:
        print(f"Downloading {repo}...")
        snapshot_download(repo_id=repo)
    print("Surya models downloaded successfully.")
