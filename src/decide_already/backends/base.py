"""The one interface every decision-model backend implements. Jev is first; OpenJev or CLM can follow."""

from typing import Protocol

from ..types import Group, StrictModel


class SortRequest(StrictModel):
    """Everything a backend needs to judge one file."""

    file_id: str
    state: dict  # what the model reads: the card, or an experiment variant of it
    groups: list[Group]
    set_version: int
    variant: str = "card"
    with_descriptions: bool = True  # False only for the names-only experiment variant


class SortResult(StrictModel):
    """A backend's judgment of one file. Code turns it into a placement (see sort.decide)."""

    file_id: str
    probs: dict[str, float]  # by group id; sums to about 1; empty when the backend has no pick
    fits_any: (
        float  # probability that some group clearly fits; 1.0 for backends that cannot abstain
    )
    model: str  # the exact model version that answered
    option_order: list[str]  # group ids in the order they were sent
    input_tokens: int | None = None
    output_tokens: int | None = None  # only for backends that bill output (the LLM baseline)
    request_id: str | None = None
    latency_ms: float | None = None


class BackendError(Exception):
    """The backend could not produce an answer. A failure is never treated as an answer."""


class BudgetExceeded(BackendError):
    """The vault's spend cap was reached, so no request was sent."""


class Backend(Protocol):
    """A decision model: reads a state, returns probabilities over the groups."""

    name: str

    def sort(self, request: SortRequest) -> SortResult:
        """Judge one file. Raises BackendError when no answer could be produced."""
        ...


def state_text(state: dict) -> str:
    """Flatten a state into plain "key: value" lines, for backends that read text instead of JSON."""
    lines: list[str] = []

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), inner)
        elif isinstance(value, list):
            if value:
                lines.append(f"{prefix}: " + "; ".join(str(item) for item in value))
        elif value not in (None, ""):
            lines.append(f"{prefix}: {value}")

    walk("", state)
    return "\n".join(lines)
