"""Embedding-similarity baseline: the card's embedding against each group's description.

The nearest group wins. The model runs on this machine through fastembed, so nothing is sent
anywhere, but the first use downloads the model files into the cache folder.
"""

import math
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ...backends.base import BackendError, SortRequest, SortResult, state_text
from ...types import Group

# Small English model, 384 dimensions. It reads at most 512 tokens and drops the rest, so of a
# 3k-token card it sees the file name, type, title, outline and the start of the opening text.
# Set chunk_chars to read the whole card in pieces, or pass another fastembed model.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# A function from a list of texts to one vector per text, in the same order.
Embed = Callable[[list[str]], Iterable[Sequence[float]]]


def group_text(group: Group, with_descriptions: bool = True, include_excludes: bool = False) -> str:
    """The text embedded for a group: its name and what it covers, or the name alone.

    What a group excludes is left out unless asked for. An embedding has no notion of "not":
    the excluded words pull a group toward the very documents it should reject.
    """
    if not with_descriptions:
        return group.name
    text = f"{group.name}: {group.covers}"
    if include_excludes and group.excludes:
        text += f". Excludes: {group.excludes}"
    return text


def chunks(text: str, size: int) -> list[str]:
    """Cut text into pieces of at most `size` characters, in order. Never returns an empty list."""
    return [text[start : start + size] for start in range(0, len(text), size)] or [""]


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    """The element-wise mean of some vectors of the same length."""
    return [sum(values) / len(vectors) for values in zip(*vectors)]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors: 1.0 for the same direction, 0.0 if either is all zeros."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def softmax(scores: Sequence[float], temperature: float) -> list[float]:
    """Turn scores into probabilities that sum to 1. A lower temperature sharpens them."""
    if not scores:
        return []
    top = max(scores)
    weights = [math.exp((score - top) / temperature) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


class EmbeddingBackend:
    """Sorts by embedding similarity alone. It cannot abstain, so fits_any is always 1.0."""

    name = "embed"

    def __init__(
        self,
        embed: Embed | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str | Path | None = None,
        temperature: float = 0.05,
        include_excludes: bool = False,
        chunk_chars: int | None = None,
    ):
        """Keep the settings. With no embed function, the fastembed model is built on first use.

        cache_dir is where fastembed keeps the downloaded model (its own default when None).
        temperature divides the similarities before the softmax.
        include_excludes adds each group's excludes text to what is embedded for it.
        chunk_chars, when set, embeds the card in pieces of that many characters and averages
        them, so a model with a short reading window still sees the whole card.
        """
        if temperature <= 0:
            raise ValueError("temperature must be above 0")
        if chunk_chars is not None and chunk_chars < 1:
            raise ValueError("chunk_chars must be at least 1")
        self.model_name = model_name
        self.temperature = temperature
        self.include_excludes = include_excludes
        self.chunk_chars = chunk_chars
        self._embed = embed
        self._cache_dir = cache_dir
        # Group vectors, kept per group set: the key is the exact texts that were embedded.
        self._group_vectors: dict[tuple[str, ...], list[list[float]]] = {}

    def sort(self, request: SortRequest) -> SortResult:
        """Embed the card text and give each group the softmax of its cosine similarity.

        latency_ms covers this file only. Loading the model and embedding the groups happen
        once per group set and are left out.
        """
        texts = tuple(
            group_text(group, request.with_descriptions, self.include_excludes)
            for group in request.groups
        )
        if texts not in self._group_vectors:
            self._group_vectors[texts] = self._vectors(list(texts))
        started = time.perf_counter()
        card_text = state_text(request.state)
        if self.chunk_chars is None:
            card_vector = self._vectors([card_text])[0]
        else:
            card_vector = mean_vector(self._vectors(chunks(card_text, self.chunk_chars)))
        similarities = [cosine(card_vector, vector) for vector in self._group_vectors[texts]]
        shares = softmax(similarities, self.temperature)
        return SortResult(
            file_id=request.file_id,
            probs={group.id: share for group, share in zip(request.groups, shares)},
            fits_any=1.0,
            model=f"embed:{self.model_name}",
            option_order=[group.id for group in request.groups],
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        """Embed the texts as plain lists of floats. Any failure becomes a BackendError."""
        try:
            if self._embed is None:
                self._embed = self._load_model()
            vectors = [[float(x) for x in vector] for vector in self._embed(texts)]
        except Exception as exc:
            raise BackendError(f"embedding failed ({self.model_name}): {exc}") from exc
        if len(vectors) != len(texts):
            raise BackendError(f"embedding returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors

    def _load_model(self) -> Embed:
        """Build the fastembed model and return its embed function."""
        # Imported here because fastembed is slow to import and only this path needs it.
        from fastembed import TextEmbedding

        cache_dir = str(self._cache_dir) if self._cache_dir is not None else None
        model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)
        return model.embed
