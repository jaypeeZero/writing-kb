import re
from collections import defaultdict

from breame.spelling import get_american_spelling
from rank_bm25 import BM25Plus

import writing_kb.content as content_module


class SearchIndex:
    """Multi-signal search index combining BM25 (full + title), token coverage, and bigram proximity via RRF."""

    def __init__(self):
        self.documents: list[dict] = []
        self.bm25_full: BM25Plus | None = None
        self.bm25_title: BM25Plus | None = None
        self._doc_token_sets: list[set[str]] = []
        self._doc_bigram_sets: list[set[tuple[str, str]]] = []
        self._indexed = False

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, extract words, normalise British/American spelling variants."""
        return [get_american_spelling(w) for w in re.findall(r"[a-z]+", text.lower())]

    def _bigrams(self, tokens: list[str]) -> set[tuple[str, str]]:
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    def build(self, sections: list[dict]):
        self.documents = sections
        if not sections:
            self._indexed = True
            return

        full = [self._tokenize(d["section_title"] + " " + d["content"]) for d in sections]
        titles = [self._tokenize(d["section_title"]) or [""] for d in sections]

        self._doc_token_sets = [set(t) for t in full]
        self._doc_bigram_sets = [self._bigrams(t) for t in full]
        self.bm25_full = BM25Plus(full)
        self.bm25_title = BM25Plus(titles)
        self._indexed = True

    def _rank_bm25(self, bm25: BM25Plus, query_tokens: list[str]) -> list[int]:
        scores = bm25.get_scores(query_tokens)
        return [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    def _rank_coverage(self, query_tokens: list[str]) -> list[int]:
        if not query_tokens:
            return list(range(len(self.documents)))
        query_set = set(query_tokens)
        scores = [len(query_set & s) / len(query_set) for s in self._doc_token_sets]
        return [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    def _rank_bigrams(self, query_tokens: list[str]) -> list[int]:
        query_bigrams = self._bigrams(query_tokens)
        if not query_bigrams:
            return list(range(len(self.documents)))
        scores = [len(query_bigrams & b) for b in self._doc_bigram_sets]
        return [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    def _rrf(self, *ranked_lists: list[int], k: int = 60) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] += 1 / (k + rank)
        return sorted(scores, key=lambda i: scores[i], reverse=True)

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        if not self._indexed or not self.documents:
            return []
        tokens = self._tokenize(query)
        query_set = set(tokens)
        fused = self._rrf(
            self._rank_bm25(self.bm25_full, tokens),
            self._rank_bm25(self.bm25_title, tokens),
            self._rank_coverage(tokens),
            self._rank_bigrams(tokens),
        )
        return [
            self.documents[i] for i in fused[:top_n]
            if query_set & self._doc_token_sets[i]
        ]


_search_index = SearchIndex()


def initialize_search_index():
    """Build the search index from all KB files."""
    all_sections = []
    for p in content_module.list_md_files():
        rel = str(p.relative_to(content_module.KB_DIR))
        text = p.read_text("utf-8")
        all_sections.extend(content_module.parse_sections(text, rel))
    _search_index.build(all_sections)
