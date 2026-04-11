"""Tests for the Writing KB MCP server."""

import pytest
from unittest.mock import patch

import server
from server import (
    _safe_path, _list_md_files, _read_file, parse_sections,
    kb_list_topics, kb_read, kb_search,
)


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


class TestListMdFiles:
    """Test that file listing is restricted to CONTENT_DIRS."""

    def test_returns_files_from_content_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / "craft").mkdir()
        (tmp_path / "craft" / "dialogue.md").write_text("content")
        files = _list_md_files()
        assert len(files) == 1
        assert files[0].name == "dialogue.md"

    def test_excludes_files_outside_content_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / "craft").mkdir()
        (tmp_path / "craft" / "dialogue.md").write_text("content")
        (tmp_path / "README.md").write_text("readme")
        (tmp_path / "server.py").write_text("code")
        files = _list_md_files()
        assert all(f.name == "dialogue.md" for f in files)

    def test_excludes_venv_and_other_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / ".venv" / "lib").mkdir(parents=True)
        (tmp_path / ".venv" / "README.md").write_text("venv")
        (tmp_path / "craft").mkdir()
        (tmp_path / "craft" / "dialogue.md").write_text("content")
        files = _list_md_files()
        assert len(files) == 1
        assert files[0].name == "dialogue.md"

    def test_recurses_within_content_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / "craft" / "advanced").mkdir(parents=True)
        (tmp_path / "craft" / "dialogue.md").write_text("content")
        (tmp_path / "craft" / "advanced" / "subtext.md").write_text("content")
        files = _list_md_files()
        names = {f.name for f in files}
        assert "dialogue.md" in names
        assert "subtext.md" in names

    def test_missing_content_dir_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft", "style"})
        (tmp_path / "craft").mkdir()
        (tmp_path / "craft" / "dialogue.md").write_text("content")
        # "style" dir doesn't exist — should not crash
        files = _list_md_files()
        assert len(files) == 1


class TestReadFile:
    """Test that file reading is restricted to CONTENT_DIRS."""

    def test_reads_file_in_content_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / "craft").mkdir()
        (tmp_path / "craft" / "dialogue.md").write_text("# Dialogue")
        assert _read_file("craft/dialogue.md") == "# Dialogue"

    def test_blocks_files_outside_content_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / "README.md").write_text("readme")
        assert _read_file("README.md") is None

    def test_blocks_server_py(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / "server.py").write_text("secret code")
        assert _read_file("server.py") is None

    def test_returns_none_for_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        assert _read_file("craft/nonexistent.md") is None

    def test_blocks_traversal_outside_kb_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path / "kb")
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / "kb").mkdir()
        (tmp_path / "secret.md").write_text("secret")
        assert _read_file("../secret.md") is None

    def test_blocks_traversal_into_disallowed_sibling(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        monkeypatch.setattr(server, "CONTENT_DIRS", {"craft"})
        (tmp_path / "style").mkdir()
        (tmp_path / "style" / "prose.md").write_text("prose")
        # "style" not in CONTENT_DIRS for this test
        assert _read_file("style/prose.md") is None


class TestKbListTopics:
    """Test topic listing and article outline."""

    @pytest.mark.asyncio
    async def test_no_path_groups_by_category(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        (tmp_path / "craft").mkdir()
        (tmp_path / "craft" / "dialogue.md").write_text("# Dialogue")
        (tmp_path / "craft" / "voice.md").write_text("# Voice")
        (tmp_path / "style").mkdir()
        (tmp_path / "style" / "rhythm.md").write_text("# Rhythm")
        result = await kb_list_topics()
        assert "## craft" in result
        assert "## style" in result
        assert "- dialogue" in result
        assert "- voice" in result
        assert "- rhythm" in result

    @pytest.mark.asyncio
    async def test_no_path_sorts_topics_alphabetically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        (tmp_path / "craft").mkdir()
        (tmp_path / "craft" / "voice.md").write_text("# Voice")
        (tmp_path / "craft" / "dialogue.md").write_text("# Dialogue")
        result = await kb_list_topics()
        craft_section = result.split("## craft")[1]
        assert craft_section.index("dialogue") < craft_section.index("voice")

    @pytest.mark.asyncio
    async def test_no_path_handles_nested_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        (tmp_path / "craft" / "dialogue").mkdir(parents=True)
        (tmp_path / "craft" / "dialogue" / "advanced.md").write_text("# Advanced")
        result = await kb_list_topics()
        assert "## craft" in result
        assert "- advanced" in result

    @pytest.mark.asyncio
    async def test_with_path_returns_section_headers(self):
        content = "## Dialogue Basics\nSome text.\n## Advanced Techniques\nMore text."
        with patch("server._read_file", return_value=content):
            result = await kb_list_topics("craft/dialogue")
        assert "## Dialogue Basics" in result
        assert "## Advanced Techniques" in result

    @pytest.mark.asyncio
    async def test_with_path_indents_h3_headers(self):
        content = "## Top Section\nText.\n### Subsection\nMore text."
        with patch("server._read_file", return_value=content):
            result = await kb_list_topics("craft/dialogue")
        assert "  ### Subsection" in result

    @pytest.mark.asyncio
    async def test_with_path_includes_filepath(self):
        content = "## Section\nText."
        with patch("server._read_file", return_value=content):
            result = await kb_list_topics("craft/dialogue")
        assert "craft/dialogue.md" in result

    @pytest.mark.asyncio
    async def test_with_path_handles_404(self):
        with patch("server._read_file", return_value=None):
            result = await kb_list_topics("nonexistent")
        assert "Error: Article not found" in result

    @pytest.mark.asyncio
    async def test_with_path_rejects_traversal(self):
        result = await kb_list_topics("../etc/passwd")
        assert "Invalid path" in result

    @pytest.mark.asyncio
    async def test_with_path_no_headers_returns_message(self):
        content = "Just plain text with no headers."
        with patch("server._read_file", return_value=content):
            result = await kb_list_topics("craft/notes")
        assert "no section headers found" in result

    @pytest.mark.asyncio
    async def test_with_path_does_not_include_body_text(self):
        content = "## Header\nThis body text should not appear."
        with patch("server._read_file", return_value=content):
            result = await kb_list_topics("craft/dialogue")
        assert "body text should not appear" not in result


class TestKbRead:
    """Test article reading."""

    @pytest.mark.asyncio
    async def test_reads_full_article(self):
        content = "# Article content"
        with patch("server._read_file", return_value=content):
            result = await kb_read("style/rhythm")
        assert result == content

    @pytest.mark.asyncio
    async def test_adds_md_extension(self):
        with patch("server._read_file", return_value="# Content") as mock_read:
            await kb_read("style/rhythm")
        mock_read.assert_called_once_with("style/rhythm.md")

    @pytest.mark.asyncio
    async def test_handles_404(self):
        with patch("server._read_file", return_value=None):
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
        with patch("server._read_file", return_value=content):
            result = await kb_read("craft/dialogue", section="Basics")
        assert "Basic content here." in result
        assert "Advanced content here." not in result

    @pytest.mark.asyncio
    async def test_section_includes_header_in_output(self):
        content = "## Basics\nSome content."
        with patch("server._read_file", return_value=content):
            result = await kb_read("craft/dialogue", section="Basics")
        assert "## Basics" in result

    @pytest.mark.asyncio
    async def test_section_match_is_case_insensitive(self):
        content = "## Dialogue Basics\nContent here."
        with patch("server._read_file", return_value=content):
            result = await kb_read("craft/dialogue", section="dialogue basics")
        assert "Content here." in result

    @pytest.mark.asyncio
    async def test_section_not_found_lists_available(self):
        content = "## Basics\nContent.\n## Advanced\nMore content."
        with patch("server._read_file", return_value=content):
            result = await kb_read("craft/dialogue", section="Nonexistent")
        assert "not found" in result
        assert "Basics" in result
        assert "Advanced" in result

    @pytest.mark.asyncio
    async def test_h3_section_uses_correct_prefix(self):
        content = "### Sub Topic\nSub content."
        with patch("server._read_file", return_value=content):
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
    async def test_initializes_index_from_files_when_not_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "KB_DIR", tmp_path)
        (tmp_path / "craft").mkdir()
        (tmp_path / "craft" / "dialogue.md").write_text("## Dialogue\nDialogue content here.")
        assert not server._search_index._indexed
        await kb_search("dialogue")
        assert server._search_index._indexed
