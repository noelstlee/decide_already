"""A stand-in backend for tests: it answers from a table, with no model and no network."""

from decide_already.backends.base import BackendError, SortRequest, SortResult

Answer = tuple[dict[str, float], float]  # (probabilities by group id, fits_any)


class FakeBackend:
    """A Backend with canned answers that remembers every request it was given."""

    name = "fake"

    def __init__(
        self,
        answers: dict[str, Answer] | None = None,
        default: Answer | None = None,
        fail: set[str] | None = None,
    ) -> None:
        """Set the answers by file id, the answer for any other file, and the file ids that fail."""
        self.answers = answers if answers is not None else {}
        self.default = default
        self.fail = fail if fail is not None else set()
        self.calls: list[SortRequest] = []

    def sort(self, request: SortRequest) -> SortResult:
        """Record the request, then answer from the table, the default, or an even spread.

        A file id in `fail` raises BackendError. With no answer and no default, every group gets
        the same probability and fits_any is 0.9. Options are reported in the order given.
        """
        self.calls.append(request)
        if request.file_id in self.fail:
            raise BackendError(f"the fake backend was told to fail for {request.file_id}")
        if request.file_id in self.answers:
            probs, fits_any = self.answers[request.file_id]
        elif self.default is not None:
            probs, fits_any = self.default
        else:
            probs = {group.id: 1 / len(request.groups) for group in request.groups}
            fits_any = 0.9
        return SortResult(
            file_id=request.file_id,
            probs=probs,
            fits_any=fits_any,
            model="fake-0",
            option_order=[group.id for group in request.groups],
        )
