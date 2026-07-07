"""Token counting and token-aware text splitting.

Wraps tiktoken with a graceful fallback so the pipeline still runs (with an
approximate token count) if tiktoken or its encoding data is unavailable.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Rough chars-per-token used only by the fallback estimator.
_FALLBACK_CHARS_PER_TOKEN = 4
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class Tokenizer:
    """Encoding-aware tokenizer with a deterministic fallback."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._enc = None
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "tiktoken encoding %r unavailable (%s); falling back to char-based "
                "token estimation. Token counts will be approximate.",
                encoding_name,
                exc,
            )

    def count(self, text: str) -> int:
        """Number of tokens in `text`."""
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN) if text else 0

    def split_to_windows(
        self, text: str, max_tokens: int, overlap_tokens: int
    ) -> list[str]:
        """Split `text` into windows of <= max_tokens with token overlap.

        Uses exact token boundaries when tiktoken is available (decoding back to
        text). Otherwise falls back to sentence-greedy packing so we never cut
        mid-word. Returns [text] unchanged if it already fits.
        """
        if not text.strip():
            return []
        if self.count(text) <= max_tokens:
            return [text]

        if self._enc is not None:
            return self._split_by_tokens(text, max_tokens, overlap_tokens)
        return self._split_by_sentences(text, max_tokens, overlap_tokens)

    # -- internals ----------------------------------------------------------

    def _split_by_tokens(
        self, text: str, max_tokens: int, overlap_tokens: int
    ) -> list[str]:
        tokens = self._enc.encode(text)
        step = max(1, max_tokens - overlap_tokens)
        windows: list[str] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + max_tokens]
            if not window:
                break
            windows.append(self._enc.decode(window).strip())
            if start + max_tokens >= len(tokens):
                break
        return [w for w in windows if w]

    def _split_by_sentences(
        self, text: str, max_tokens: int, overlap_tokens: int
    ) -> list[str]:
        sentences = _SENTENCE_SPLIT.split(text)
        windows: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for sent in sentences:
            st = self.count(sent)
            if current and current_tokens + st > max_tokens:
                windows.append(" ".join(current).strip())
                # carry overlap sentences from the tail
                current, current_tokens = self._tail_overlap(current, overlap_tokens)
            current.append(sent)
            current_tokens += st
        if current:
            windows.append(" ".join(current).strip())
        return [w for w in windows if w]

    def _tail_overlap(
        self, sentences: list[str], overlap_tokens: int
    ) -> tuple[list[str], int]:
        kept: list[str] = []
        total = 0
        for sent in reversed(sentences):
            st = self.count(sent)
            if total + st > overlap_tokens:
                break
            kept.insert(0, sent)
            total += st
        return kept, total

    def tail_text(self, text: str, overlap_tokens: int) -> str:
        """Return the last ~overlap_tokens worth of `text` (for chunk overlap)."""
        if overlap_tokens <= 0 or not text:
            return ""
        if self._enc is not None:
            tokens = self._enc.encode(text)
            if len(tokens) <= overlap_tokens:
                return text
            return self._enc.decode(tokens[-overlap_tokens:]).strip()
        # fallback: approximate by characters
        approx_chars = overlap_tokens * _FALLBACK_CHARS_PER_TOKEN
        return text[-approx_chars:].strip()
