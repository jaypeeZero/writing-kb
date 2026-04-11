"""Tests for the Writing KB MCP server."""

import pytest
from base64 import b64encode
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import server
from server import _safe_path, _list_files, kb_list_topics, kb_read, kb_search


def make_mock_response(json_data, status_code=200):
    """Create a properly configured mock HTTP response."""
    mock = AsyncMock()
    mock.status_code = status_code
    mock.json = MagicMock(return_value=json_data)
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=mock
        )
    return mock


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


class TestListFiles:
    """Test GitHub API file listing."""

    @pytest.mark.asyncio
    async def test_lists_md_files_only(self):
        mock_response = make_mock_response([
            {"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"},
            {"name": "voice.md", "type": "file", "path": "craft/voice.md"},
            {"name": "notes.txt", "type": "file", "path": "craft/notes.txt"},
        ])

        with patch("server._github_get", return_value=mock_response):
            files = await _list_files("craft")

        assert len(files) == 2
        assert all(f["name"].endswith(".md") for f in files)

    @pytest.mark.asyncio
    async def test_excludes_configured_files(self):
        mock_response = make_mock_response([
            {"name": "README.md", "type": "file", "path": "README.md"},
            {"name": "server.py", "type": "file", "path": "server.py"},
            {"name": "prose.md", "type": "file", "path": "style/prose.md"},
        ])

        with patch("server._github_get", return_value=mock_response):
            files = await _list_files()

        assert len(files) == 1
        assert files[0]["name"] == "prose.md"

    @pytest.mark.asyncio
    async def test_excludes_hidden_files(self):
        mock_response = make_mock_response([
            {"name": ".gitignore", "type": "file", "path": ".gitignore"},
            {"name": ".hidden.md", "type": "file", "path": ".hidden.md"},
            {"name": "visible.md", "type": "file", "path": "visible.md"},
        ])

        with patch("server._github_get", return_value=mock_response):
            files = await _list_files()

        assert len(files) == 1
        assert files[0]["name"] == "visible.md"

    @pytest.mark.asyncio
    async def test_recurses_into_directories(self):
        mock_response = make_mock_response(None)
        mock_response.json.side_effect = [
            [
                {"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"},
                {"name": "advanced", "type": "dir", "path": "craft/advanced"},
            ],
            [
                {"name": "subtext.md", "type": "file", "path": "craft/advanced/subtext.md"},
            ],
        ]

        with patch("server._github_get", return_value=mock_response):
            files = await _list_files("craft")

        assert len(files) == 2
        paths = {f["path"] for f in files}
        assert "craft/dialogue.md" in paths
        assert "craft/advanced/subtext.md" in paths


class TestKbListTopics:
    """Test topic listing by category."""

    @pytest.mark.asyncio
    async def test_groups_by_category(self):
        mock_files = [
            {"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"},
            {"name": "voice.md", "type": "file", "path": "craft/voice.md"},
            {"name": "rhythm.md", "type": "file", "path": "style/rhythm.md"},
        ]

        with patch("server._list_files", return_value=mock_files):
            result = await kb_list_topics()

        assert "## craft" in result
        assert "## style" in result
        assert "- dialogue" in result
        assert "- voice" in result
        assert "- rhythm" in result

    @pytest.mark.asyncio
    async def test_sorts_topics_alphabetically(self):
        mock_files = [
            {"name": "voice.md", "type": "file", "path": "craft/voice.md"},
            {"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"},
        ]

        with patch("server._list_files", return_value=mock_files):
            result = await kb_list_topics()

        craft_section = result.split("## craft")[1].split("##")[0] if "##" in result.split("## craft")[1] else result.split("## craft")[1]
        assert craft_section.index("dialogue") < craft_section.index("voice")

    @pytest.mark.asyncio
    async def test_handles_nested_paths(self):
        mock_files = [
            {"name": "advanced.md", "type": "file", "path": "craft/dialogue/advanced.md"},
        ]

        with patch("server._list_files", return_value=mock_files):
            result = await kb_list_topics()

        assert "## craft" in result
        assert "- advanced" in result


class TestKbRead:
    """Test article reading."""

    @pytest.mark.asyncio
    async def test_reads_article(self):
        content = b64encode(b"# Article content").decode()
        mock_response = make_mock_response({"content": content})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_read("style/rhythm")

        assert result == "# Article content"

    @pytest.mark.asyncio
    async def test_adds_md_extension(self):
        content = b64encode(b"# Content").decode()
        mock_response = make_mock_response({"content": content})

        with patch("server._github_get", return_value=mock_response) as mock_get:
            await kb_read("style/rhythm")

        called_path = mock_get.call_args[0][0]
        assert "style/rhythm.md" in called_path

    @pytest.mark.asyncio
    async def test_handles_404(self):
        mock_response = make_mock_response({"message": "Not Found"}, status_code=404)

        with patch("server._github_get", return_value=mock_response):
            result = await kb_read("nonexistent")

        assert "Error: Article not found" in result

    @pytest.mark.asyncio
    async def test_rejects_traversal(self):
        result = await kb_read("../etc/passwd")
        assert "Invalid path" in result

    @pytest.mark.asyncio
    async def test_rejects_backslash(self):
        result = await kb_read("style\\..\\..\\etc")
        assert "Invalid path" in result

    @pytest.mark.asyncio
    async def test_handles_empty_path(self):
        result = await kb_read("")
        assert "Empty path" in result


class TestKbSearch:
    """Test article search functionality."""

    @pytest.mark.asyncio
    async def test_finds_matching_article(self):
        content = b64encode(b"# Dialogue tips\nGood dialogue shows character.").decode()
        mock_response = make_mock_response({"content": content})
        mock_files = [
            {"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"},
        ]

        with patch("server._list_files", return_value=mock_files):
            with patch("server._github_get", return_value=mock_response):
                result = await kb_search("dialogue")

        assert "craft/dialogue.md" in result

    @pytest.mark.asyncio
    async def test_case_insensitive_search(self):
        content = b64encode(b"# DIALOGUE tips").decode()
        mock_response = make_mock_response({"content": content})
        mock_files = [
            {"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"},
        ]

        with patch("server._list_files", return_value=mock_files):
            with patch("server._github_get", return_value=mock_response):
                result = await kb_search("DIALOGUE")

        assert "craft/dialogue.md" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        mock_files = [
            {"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"},
        ]
        mock_response = make_mock_response({"content": b64encode(b"# Other content").decode()})

        with patch("server._list_files", return_value=mock_files):
            with patch("server._github_get", return_value=mock_response):
                result = await kb_search("nonexistent")

        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_limits_context_lines(self):
        lines = [f"Line {i}: contains target" for i in range(10)]
        content = b64encode("\n".join(lines).encode()).decode()
        mock_response = make_mock_response({"content": content})
        mock_files = [
            {"name": "article.md", "type": "file", "path": "article.md"},
        ]

        with patch("server._list_files", return_value=mock_files):
            with patch("server._github_get", return_value=mock_response):
                result = await kb_search("target")

        match_count = result.count("contains target")
        assert match_count <= 3

    @pytest.mark.asyncio
    async def test_handles_api_failure_gracefully(self):
        mock_files = [
            {"name": "article.md", "type": "file", "path": "article.md"},
        ]
        mock_response = make_mock_response({"message": "Error"}, status_code=500)

        with patch("server._list_files", return_value=mock_files):
            with patch("server._github_get", return_value=mock_response):
                result = await kb_search("query")

        assert result == "No results found."
