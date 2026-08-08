"""Unit tests for device detection/validation in ``rvc_assets.detect_device``.

``rvc_assets`` does ``import torch`` at module level, so its ``torch`` binding
is fixed at import time and sys.modules mocking cannot affect it.  We therefore
patch attributes on the already-imported module (``patch.object(rvc_assets.torch,
...)``) — the robust approach regardless of import order — and, for the "no xpu
attribute" case, swap the module-level ``torch`` reference for a minimal fake.

No GPU-dependent e2e tests: every device probe is mocked.
"""

import pytest
import torch as _real_torch
from types import SimpleNamespace
from unittest.mock import patch

import kokoclone_train.core.rvc_assets as rvc_assets

from kokoclone_train.core.rvc_assets import detect_device, empty_cache


def _fake_torch(*, has_xpu: bool = True, has_mps_cache: bool = True) -> SimpleNamespace:
    """Minimal torch-like object; ``device`` delegates to the real torch.

    ``has_xpu=False`` omits the ``xpu`` attribute entirely (``hasattr`` must
    be False, so ``None`` is not acceptable).
    """
    attrs = {
        "device": _real_torch.device,
        "backends": SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        ),
        "cuda": SimpleNamespace(is_available=lambda: False),
        "mps": (
            SimpleNamespace(empty_cache=lambda: None)
            if has_mps_cache
            else SimpleNamespace()
        ),
    }
    if has_xpu:
        attrs["xpu"] = SimpleNamespace(is_available=lambda: True)
    return SimpleNamespace(**attrs)


# ── Auto-detect (no-arg / "auto") regression ──────────────────────────────

def test_detect_auto_no_arg():
    """No-arg call with nothing available falls back to cpu."""
    with (
        patch.object(rvc_assets.torch.backends.mps, "is_available", return_value=False),
        patch.object(rvc_assets.torch.cuda, "is_available", return_value=False),
        patch.object(rvc_assets.torch.xpu, "is_available", return_value=False),
    ):
        assert detect_device() == _real_torch.device("cpu")
        assert detect_device("auto") == _real_torch.device("cpu")


def test_detect_auto_mps_first():
    """MPS wins the auto chain when available (order preserved)."""
    with patch.object(rvc_assets.torch.backends.mps, "is_available", return_value=True):
        assert detect_device() == _real_torch.device("mps")


def test_detect_auto_cuda():
    """CUDA beats XPU in the auto chain (mps unavailable)."""
    with (
        patch.object(rvc_assets.torch.backends.mps, "is_available", return_value=False),
        patch.object(rvc_assets.torch.cuda, "is_available", return_value=True),
        patch.object(rvc_assets.torch.xpu, "is_available", return_value=True),
    ):
        assert detect_device() == _real_torch.device("cuda")


def test_detect_auto_xpu():
    """XPU chosen when mps+cuda unavailable."""
    with (
        patch.object(rvc_assets.torch.backends.mps, "is_available", return_value=False),
        patch.object(rvc_assets.torch.cuda, "is_available", return_value=False),
        patch.object(rvc_assets.torch.xpu, "is_available", return_value=True),
    ):
        assert detect_device() == _real_torch.device("xpu")


# ── Explicit requests ─────────────────────────────────────────────────────

def test_detect_cpu():
    """cpu always succeeds regardless of availability."""
    with patch.object(rvc_assets.torch.cuda, "is_available", return_value=True):
        assert detect_device("cpu") == _real_torch.device("cpu")


def test_detect_cuda_available():
    """Bare 'cuda' resolves to an unindexed cuda device."""
    with (
        patch.object(rvc_assets.torch.cuda, "is_available", return_value=True),
        patch.object(rvc_assets.torch.cuda, "device_count", return_value=1),
    ):
        dev = detect_device("cuda")
        assert dev.type == "cuda"
        assert dev.index is None


def test_detect_cuda_indexed():
    """'cuda:1' resolves to an explicit indexed device."""
    with (
        patch.object(rvc_assets.torch.cuda, "is_available", return_value=True),
        patch.object(rvc_assets.torch.cuda, "device_count", return_value=2),
    ):
        dev = detect_device("cuda:1")
        assert dev == _real_torch.device("cuda:1")


def test_detect_cuda_unavailable_raises():
    """Requesting cuda with no CUDA build raises an actionable ValueError."""
    with (
        patch.object(rvc_assets.torch.backends.mps, "is_available", return_value=False),
        patch.object(rvc_assets.torch.cuda, "is_available", return_value=False),
        patch.object(rvc_assets.torch.xpu, "is_available", return_value=False),
    ):
        with pytest.raises(ValueError) as excinfo:
            detect_device("cuda:0")
        msg = str(excinfo.value)
        assert "cuda" in msg
        assert "ROCm" in msg
        assert "ROCM_RX470_SETUP.md" in msg
        assert "auto" in msg


def test_detect_cuda_index_oob_raises():
    """cuda:2 with one device reports the device count."""
    with (
        patch.object(rvc_assets.torch.cuda, "is_available", return_value=True),
        patch.object(rvc_assets.torch.cuda, "device_count", return_value=1),
    ):
        with pytest.raises(ValueError) as excinfo:
            detect_device("cuda:2")
        assert "1" in str(excinfo.value)
        assert "cuda:2" in str(excinfo.value)


def test_detect_xpu_available():
    """'xpu' resolves when the XPU backend is available."""
    with (
        patch.object(rvc_assets.torch.xpu, "is_available", return_value=True),
        patch.object(rvc_assets.torch.xpu, "device_count", return_value=1),
    ):
        dev = detect_device("xpu")
        assert dev.type == "xpu"
        assert dev.index is None


def test_detect_xpu_indexed():
    """'xpu:1' resolves to an explicit indexed device."""
    with (
        patch.object(rvc_assets.torch.xpu, "is_available", return_value=True),
        patch.object(rvc_assets.torch.xpu, "device_count", return_value=2),
    ):
        dev = detect_device("xpu:1")
        assert dev == _real_torch.device("xpu:1")


def test_detect_xpu_unavailable_raises():
    """XPU build present but no device available raises."""
    with patch.object(rvc_assets.torch.xpu, "is_available", return_value=False):
        with pytest.raises(ValueError) as excinfo:
            detect_device("xpu")
        assert "xpu" in str(excinfo.value)


def test_detect_xpu_no_attr_raises():
    """torch build without an xpu attribute raises."""
    with patch.object(rvc_assets, "torch", _fake_torch(has_xpu=False)):
        with pytest.raises(ValueError) as excinfo:
            detect_device("xpu")
        assert "xpu" in str(excinfo.value)


def test_detect_mps_available():
    """'mps' resolves when the MPS backend is available."""
    with patch.object(rvc_assets.torch.backends.mps, "is_available", return_value=True):
        assert detect_device("mps") == _real_torch.device("mps")


def test_detect_mps_unavailable_raises():
    """'mps' on a non-Apple build raises."""
    with patch.object(rvc_assets.torch.backends.mps, "is_available", return_value=False):
        with pytest.raises(ValueError) as excinfo:
            detect_device("mps")
        assert "MPS" in str(excinfo.value)


def test_detect_unknown_raises():
    """Unknown device name raises with the valid values listed."""
    with pytest.raises(ValueError) as excinfo:
        detect_device("foo")
    msg = str(excinfo.value)
    for valid in ("auto", "cuda", "xpu", "mps", "cpu"):
        assert valid in msg


def test_detect_bad_index_raises():
    """Non-integer index (cuda:abc) raises a clear error."""
    with pytest.raises(ValueError) as excinfo:
        detect_device("cuda:abc")
    assert "cuda:abc" in str(excinfo.value)


def test_detect_case_whitespace():
    """Input is case/whitespace normalized."""
    with (
        patch.object(rvc_assets.torch.cuda, "is_available", return_value=True),
        patch.object(rvc_assets.torch.cuda, "device_count", return_value=1),
    ):
        assert detect_device("  CUDA:0 ") == _real_torch.device("cuda:0")


# ── empty_cache ───────────────────────────────────────────────────────────

def test_empty_cache_cpu_noop():
    """cpu is a no-op and never touches framework caches."""
    with (
        patch.object(rvc_assets.torch.cuda, "empty_cache") as m_cuda,
        patch.object(rvc_assets.torch.xpu, "empty_cache") as m_xpu,
    ):
        empty_cache(_real_torch.device("cpu"))
        m_cuda.assert_not_called()
        m_xpu.assert_not_called()


def test_empty_cache_mps():
    """mps calls torch.mps.empty_cache when it exists."""
    with patch.object(rvc_assets.torch.mps, "empty_cache") as m_mps:
        empty_cache(_real_torch.device("mps"))
        m_mps.assert_called_once_with()


def test_empty_cache_mps_guarded():
    """mps without torch.mps.empty_cache is a silent no-op."""
    with patch.object(rvc_assets, "torch", _fake_torch(has_mps_cache=False)):
        empty_cache(_real_torch.device("mps"))  # must not raise


def test_empty_cache_cuda():
    """cuda calls torch.cuda.empty_cache."""
    with patch.object(rvc_assets.torch.cuda, "empty_cache") as m_cuda:
        empty_cache(_real_torch.device("cuda"))
        m_cuda.assert_called_once_with()


def test_empty_cache_xpu():
    """xpu calls torch.xpu.empty_cache."""
    with patch.object(rvc_assets.torch.xpu, "empty_cache") as m_xpu:
        empty_cache(_real_torch.device("xpu"))
        m_xpu.assert_called_once_with()
