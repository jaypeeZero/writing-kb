"""Tests for the Writing KB MCP server."""

import pytest
from server import _safe_path


class TestSafePath:
    """Test path validation and normalization."""

    def test_valid_simple_path(self):
        assert _safe_path("style/prose-rhythm") == "style/prose-rhythm"

    def test_valid_nested_path(self):
        assert _safe_path("craft/dialogue/advanced") == "craft/dialogue/advanced"

    def test_adds_slashes(self):
        assert _safe_path("style") == "style"

    def test_normalizes_leading_slash(self):
        assert _safe_path("/style/prose") == "style/prose"

    def test_normalizes_double_slashes(self):
        assert _safe_path("style//prose") == "style/prose"

    def test_normalizes_trailing_slash(self):
        assert _safe_path("style/prose/") == "style/prose"

    def test_rejects_parent_traversal(self):
        with pytest.raises(ValueError, match="Invalid path"):
            _safe_path("../etc/passwd")

    def test_rejects_dotdot_in_path(self):
        with pytest.raises(ValueError, match="Invalid path"):
            _safe_path("style/../../../etc/passwd")

    def test_normalizes_absolute_path(self):
        # Absolute paths get normalized (leading slash stripped)
        assert _safe_path("/etc/passwd") == "etc/passwd"

    def test_rejects_backslashes(self):
        with pytest.raises(ValueError, match="Invalid path"):
            _safe_path("style\\..\\..\\etc\\passwd")

    def test_rejects_empty_path(self):
        with pytest.raises(ValueError, match="Empty path"):
            _safe_path("")

    def test_rejects_only_slashes(self):
        with pytest.raises(ValueError, match="Empty path"):
            _safe_path("///")

    def test_valid_path_with_md_extension(self):
        assert _safe_path("style/prose.md") == "style/prose.md"
