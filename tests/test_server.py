"""Tests for the Writing KB MCP server."""

import pytest
from base64 import b64encode
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import server
from server import (
    _safe_path, _list_files, parse_sections,
    kb_list_topics, kb_read, kb_search,
)


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


def encoded(text: str) -> str:
    return b64encode(text.encode()).decode()


@pytest.fixture(autouse=True)
def reset_search_index():
    """Reset the search index between tests to prevent cross-test contamination."""
    server._search_index = server.SearchIndex()
    yield
    server._search_index = server.SearchIndex()


def make_sections(*args) -> list[dict]:
    """Build section dicts for pre-loading the search index. Args: (filepath, title, content) triples."""
    sections = []
    it = iter(args)
    for filepath, title, content in zip(it, it, it):
        sections.append({"filepath": filepath, "section_title": title, "level": 2, "content": content})
    return sections


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


class TestParseSections:
    """Test markdown section parsing."""

    def test_no_headers_returns_intro_section(self):
        content = "Some introductory text\nwith multiple lines."
        sections = parse_sections(content, "test.md")
        assert len(sections) == 1
        assert sections[0]["section_title"] == "Introduction"
        assert sections[0]["filepath"] == "test.md"

    def test_h2_headers_split_content(self):
        content = "## First\nFirst content.\n## Second\nSecond content."
        sections = parse_sections(content, "test.md")
        assert len(sections) == 2
        assert sections[0]["section_title"] == "First"
        assert sections[0]["content"] == "First content."
        assert sections[1]["section_title"] == "Second"
        assert sections[1]["content"] == "Second content."

    def test_h3_headers_split_content(self):
        content = "### Sub A\nContent A.\n### Sub B\nContent B."
        sections = parse_sections(content, "test.md")
        assert len(sections) == 2
        assert sections[0]["section_title"] == "Sub A"
        assert sections[1]["section_title"] == "Sub B"

    def test_mixed_h2_and_h3_headers(self):
        content = "## Top\nTop content.\n### Sub\nSub content."
        sections = parse_sections(content, "test.md")
        assert len(sections) == 2
        assert sections[0]["section_title"] == "Top"
        assert sections[1]["section_title"] == "Sub"

    def test_empty_sections_filtered_out(self):
        content = "## Empty\n## Non-empty\nHas content."
        sections = parse_sections(content, "test.md")
        assert len(sections) == 1
        assert sections[0]["section_title"] == "Non-empty"

    def test_content_is_stripped(self):
        content = "## Header\n\n  some text  \n\n"
        sections = parse_sections(content, "test.md")
        assert sections[0]["content"] == "some text"

    def test_section_level_recorded(self):
        content = "## H2\nContent.\n### H3\nMore."
        sections = parse_sections(content, "test.md")
        assert sections[0]["level"] == 2
        assert sections[1]["level"] == 3

    def test_filepath_preserved_in_all_sections(self):
        content = "## A\nContent A.\n## B\nContent B."
        sections = parse_sections(content, "craft/dialogue.md")
        assert all(s["filepath"] == "craft/dialogue.md" for s in sections)

    def test_empty_content_returns_empty_list(self):
        sections = parse_sections("", "test.md")
        assert sections == []


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
    """Test topic listing and article outline."""

    @pytest.mark.asyncio
    async def test_no_path_groups_by_category(self):
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
    async def test_no_path_sorts_topics_alphabetically(self):
        mock_files = [
            {"name": "voice.md", "type": "file", "path": "craft/voice.md"},
            {"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"},
        ]

        with patch("server._list_files", return_value=mock_files):
            result = await kb_list_topics()

        craft_section = result.split("## craft")[1]
        assert craft_section.index("dialogue") < craft_section.index("voice")

    @pytest.mark.asyncio
    async def test_no_path_handles_nested_paths(self):
        mock_files = [
            {"name": "advanced.md", "type": "file", "path": "craft/dialogue/advanced.md"},
        ]

        with patch("server._list_files", return_value=mock_files):
            result = await kb_list_topics()

        assert "## craft" in result
        assert "- advanced" in result

    @pytest.mark.asyncio
    async def test_with_path_returns_section_headers(self):
        content = "## Dialogue Basics\nSome text.\n## Advanced Techniques\nMore text."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_list_topics("craft/dialogue")

        assert "## Dialogue Basics" in result
        assert "## Advanced Techniques" in result

    @pytest.mark.asyncio
    async def test_with_path_indents_h3_headers(self):
        content = "## Top Section\nText.\n### Subsection\nMore text."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_list_topics("craft/dialogue")

        assert "  ### Subsection" in result

    @pytest.mark.asyncio
    async def test_with_path_includes_filepath(self):
        content = "## Section\nText."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_list_topics("craft/dialogue")

        assert "craft/dialogue.md" in result

    @pytest.mark.asyncio
    async def test_with_path_handles_404(self):
        mock_response = make_mock_response({"message": "Not Found"}, status_code=404)

        with patch("server._github_get", return_value=mock_response):
            result = await kb_list_topics("nonexistent")

        assert "Error: Article not found" in result

    @pytest.mark.asyncio
    async def test_with_path_rejects_traversal(self):
        result = await kb_list_topics("../etc/passwd")
        assert "Invalid path" in result

    @pytest.mark.asyncio
    async def test_with_path_no_headers_returns_message(self):
        content = "Just plain text with no headers."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_list_topics("craft/notes")

        assert "no section headers found" in result

    @pytest.mark.asyncio
    async def test_with_path_does_not_include_body_text(self):
        content = "## Header\nThis body text should not appear."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_list_topics("craft/dialogue")

        assert "body text should not appear" not in result


class TestKbRead:
    """Test article reading."""

    @pytest.mark.asyncio
    async def test_reads_full_article(self):
        content = "# Article content"
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_read("style/rhythm")

        assert result == content

    @pytest.mark.asyncio
    async def test_adds_md_extension(self):
        mock_response = make_mock_response({"content": encoded("# Content")})

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

    @pytest.mark.asyncio
    async def test_section_returns_only_that_section(self):
        content = "## Basics\nBasic content here.\n## Advanced\nAdvanced content here."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_read("craft/dialogue", section="Basics")

        assert "Basic content here." in result
        assert "Advanced content here." not in result

    @pytest.mark.asyncio
    async def test_section_includes_header_in_output(self):
        content = "## Basics\nSome content."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_read("craft/dialogue", section="Basics")

        assert "## Basics" in result

    @pytest.mark.asyncio
    async def test_section_match_is_case_insensitive(self):
        content = "## Dialogue Basics\nContent here."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_read("craft/dialogue", section="dialogue basics")

        assert "Content here." in result

    @pytest.mark.asyncio
    async def test_section_not_found_lists_available(self):
        content = "## Basics\nContent.\n## Advanced\nMore content."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_read("craft/dialogue", section="Nonexistent")

        assert "not found" in result
        assert "Basics" in result
        assert "Advanced" in result

    @pytest.mark.asyncio
    async def test_h3_section_uses_correct_prefix(self):
        content = "### Sub Topic\nSub content."
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._github_get", return_value=mock_response):
            result = await kb_read("craft/dialogue", section="Sub Topic")

        assert result.startswith("### Sub Topic")


class TestNormalize:
    """Test spelling normalization in tokenizer."""

    def setup_method(self):
        self.idx = server.SearchIndex()

    def tokenize(self, text):
        return self.idx._tokenize(text)

    def test_dialogue_normalizes_to_dialog(self):
        assert "dialog" in self.tokenize("dialogue")

    def test_dialog_unchanged(self):
        assert "dialog" in self.tokenize("dialog")

    def test_colour_normalizes_to_color(self):
        assert "color" in self.tokenize("colour")

    def test_recognise_normalizes_to_recognize(self):
        assert "recognize" in self.tokenize("recognise")

    def test_behaviour_normalizes_to_behavior(self):
        assert "behavior" in self.tokenize("behaviour")

    def test_monologue_normalizes_to_monolog(self):
        assert "monolog" in self.tokenize("monologue")

    def test_american_spelling_unchanged(self):
        assert "color" in self.tokenize("color")
        assert "recognize" in self.tokenize("recognize")

    def test_query_and_content_normalize_identically(self):
        """dialogue in query and dialog in content should produce same token."""
        assert self.tokenize("dialogue") == self.tokenize("dialog")

    def test_mixed_query(self):
        tokens = self.tokenize("writing dialogue in colour")
        assert "dialog" in tokens
        assert "color" in tokens
        assert "writing" in tokens


class TestSearchIndex:
    """Unit tests for the multi-signal SearchIndex."""

    def test_search_returns_matching_sections(self):
        idx = server.SearchIndex()
        idx.build(make_sections("craft/dialogue.md", "Dialogue Tips", "Good dialogue shows character through word choice."))
        results = idx.search("dialogue")
        assert len(results) == 1
        assert results[0]["section_title"] == "Dialogue Tips"

    def test_spelling_variant_matches(self):
        """'dialogue' in index, 'dialog' in query — should still match."""
        idx = server.SearchIndex()
        idx.build(make_sections("craft/dialogue.md", "Dialogue Tips", "Writing dialogue effectively."))
        results = idx.search("dialog")
        assert len(results) == 1

    def test_title_signal_ranks_title_match_above_body_match(self):
        """Section whose title matches the query should rank above one where only body matches."""
        idx = server.SearchIndex()
        idx.build(make_sections(
            "craft/craft.md", "Dialogue in Combat", "Some general writing advice.",
            "craft/craft.md", "Pacing", "Dialogue can appear in many contexts including combat scenes.",
        ))
        results = idx.search("dialogue combat")
        assert results[0]["section_title"] == "Dialogue in Combat"

    def test_bigram_signal_rewards_phrase_proximity(self):
        """Section containing adjacent query words should rank above one with scattered matches."""
        idx = server.SearchIndex()
        idx.build(make_sections(
            "f.md", "A", "Tense combat dialogue crackles with urgency.",       # bigram: combat+dialogue adjacent
            "f.md", "B", "Combat scenes are intense. Dialogue can reveal character.",  # combat and dialogue separated
        ))
        results = idx.search("combat dialogue")
        assert results[0]["section_title"] == "A"

    def test_coverage_penalises_partial_match(self):
        """Section matching all query tokens should rank above one matching only some."""
        idx = server.SearchIndex()
        idx.build(make_sections(
            "f.md", "Full Match", "Writing pacing and dialogue together creates tension.",
            "f.md", "Partial", "Dialogue is important in fiction.",
        ))
        results = idx.search("pacing dialogue")
        assert results[0]["section_title"] == "Full Match"

    def test_unmatched_query_returns_empty(self):
        idx = server.SearchIndex()
        idx.build(make_sections("f.md", "Section", "Some writing content here."))
        results = idx.search("xyzzy")
        assert results == []

    def test_build_empty_corpus_does_not_crash(self):
        idx = server.SearchIndex()
        idx.build([])
        assert idx._indexed is True
        assert idx.search("anything") == []

    def test_respects_top_n(self):
        idx = server.SearchIndex()
        idx.build(make_sections(
            "f.md", "A", "dialogue content",
            "f.md", "B", "dialogue content",
            "f.md", "C", "dialogue content",
            "f.md", "D", "dialogue content",
            "f.md", "E", "dialogue content",
            "f.md", "F", "dialogue content",
        ))
        assert len(idx.search("dialogue", top_n=3)) <= 3


class TestKbSearch:
    """Test the kb_search MCP tool."""

    def _load_index(self, *section_args):
        server._search_index.build(make_sections(*section_args))

    @pytest.mark.asyncio
    async def test_finds_matching_section(self):
        self._load_index("craft/dialogue.md", "Dialogue Tips", "Good dialogue shows character.")
        result = await kb_search("dialogue")
        assert "craft/dialogue.md" in result
        assert "Dialogue Tips" in result

    @pytest.mark.asyncio
    async def test_returns_full_section_content(self):
        body = "Dialogue reveals character. It moves plot forward. Every line should do work."
        self._load_index("craft/dialogue.md", "Writing Dialogue", body)
        result = await kb_search("dialogue")
        assert "Dialogue reveals character." in result
        assert "Every line should do work." in result

    @pytest.mark.asyncio
    async def test_spelling_variant_in_query(self):
        """Searching 'dialog' should find sections written with 'dialogue'."""
        self._load_index("craft/dialogue.md", "Dialogue Tips", "Writing dialogue effectively.")
        result = await kb_search("dialog tips")
        assert "craft/dialogue.md" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        self._load_index("craft/dialogue.md", "Other", "Unrelated content.")
        result = await kb_search("xyzzy")
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_results_separated_by_divider(self):
        self._load_index(
            "craft/combat.md", "Section A", "Combat dialogue is tense.",
            "craft/combat.md", "Section B", "More combat dialogue here.",
        )
        result = await kb_search("combat dialogue")
        assert "---" in result

    @pytest.mark.asyncio
    async def test_empty_index_returns_no_results(self):
        server._search_index.build([])
        result = await kb_search("query")
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_initializes_index_from_github_when_not_loaded(self):
        content = "## Dialogue\nDialogue content here."
        mock_files = [{"name": "dialogue.md", "type": "file", "path": "craft/dialogue.md"}]
        mock_response = make_mock_response({"content": encoded(content)})

        with patch("server._list_files", return_value=mock_files):
            with patch("server._github_get", return_value=mock_response):
                assert not server._search_index._indexed
                await kb_search("dialogue")
                assert server._search_index._indexed
