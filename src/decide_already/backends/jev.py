"""The Jev backend: the only code that talks to TypeSafe.

One request per file carries the two questions from questions.py. Before anything is sent, the
vault's spend so far is compared with the cap. Every call, good or bad, becomes one line in the
vault's log. The API key travels only in the request's Authorization header, which is never logged.
"""

import random
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx2
from typesafe_sdk import RetryPolicy, SystemOneResponse, TypeSafeClient, TypeSafeError

from ..config import Config
from ..log import append_jsonl, total_input_tokens
from ..questions import FITS_ANY_QUESTION, GROUP_QUESTION, build_questions
from ..types import Group
from .base import BackendError, BudgetExceeded, SortRequest, SortResult

API_KEY_ENV = "TYPESAFE_API_KEY"
REQUEST_ID_HEADER = "x-typesafe-request-id"
TIMEOUT_SECONDS = 30.0  # for each HTTP operation of one attempt
# Rate limits (429), request timeouts (408) and server errors (5xx) are retried with backoff,
# waiting as long as a Retry-After header asks, for at most 4 retries or 60 seconds in all.
RETRY_POLICY = RetryPolicy(
    max_retries=4,
    http_statuses={408, 429, *range(500, 600)},
    respect_retry_after=True,
    timeout=60.0,
)


class JevBackend:
    """Asks Jev about one file per call. One instance can be shared by several threads.

    The spend cap is checked before each call, so calls already under way when it is reached
    still finish.
    """

    name = "jev"

    def __init__(
        self,
        config: Config,
        log_path: Path,
        *,
        api_key: str | None = None,
        rng: random.Random | None = None,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        """Build the client and read from the log what this vault has already spent.

        The key must be passed in; make_backend looks it up. `rng` shuffles the options, so pass
        a seeded one for a reproducible order. `transport` replaces the network, for tests only.
        """
        api_key = (api_key or "").strip()
        if not api_key:
            raise BackendError(
                f"no TypeSafe API key found: set {API_KEY_ENV} in the environment or in a .env file"
            )
        self._config = config
        self._log_path = log_path
        self._api_key = api_key
        self._rng = rng if rng is not None else random.Random()
        self._lock = threading.Lock()  # guards the rng, the spend counter and the log append
        self._input_tokens = total_input_tokens(log_path)
        try:
            self._client = TypeSafeClient(
                api_key=api_key, retry=RETRY_POLICY, timeout=TIMEOUT_SECONDS, transport=transport
            )
        except TypeSafeError as error:
            raise BackendError(f"the TypeSafe client could not be set up: {error}") from error

    def sort(self, request: SortRequest) -> SortResult:
        """Judge one file: send its state with the two questions, log the call, return the answer.

        Raises BudgetExceeded, without sending, once the spend cap is reached. Raises BackendError
        when the call fails or its answer cannot be used; that call is logged with status "error".
        """
        with self._lock:
            self._check_budget()
            questions, option_order = build_questions(
                request.groups, self._rng, request.with_descriptions
            )
        payload = {"model": self._config.model, "state": request.state, "questions": questions}
        record = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "file_id": request.file_id,
            "set_version": request.set_version,
            "variant": request.variant,
            "backend": self.name,
            "model_requested": self._config.model,
            "model_answered": None,
            "option_order": option_order,
            "request": payload,
            "response": None,
            "input_tokens": None,
            "request_id": None,
            "latency_ms": None,
            "status": "error",
            "error": None,
        }
        started = time.perf_counter()
        try:
            response = self._client.system_one(
                state=payload["state"], questions=payload["questions"], model=payload["model"]
            )
        except TypeSafeError as error:
            response = None
            record["error"] = self._without_key(f"{type(error).__name__}: {error}")
            record["request_id"] = getattr(error, "request_id", None)
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if response is not None:
            raw = response.raw_http_response
            record["response"] = raw.json()
            record["request_id"] = raw.headers.get(REQUEST_ID_HEADER)
            record["model_answered"] = response.model
            record["input_tokens"] = response.usage.input_tokens
            record["error"] = self._problem(response, request.groups)
        if record["error"] is None:
            record["status"] = "ok"
        with self._lock:
            # Every answered call is billed, usable or not. total_input_tokens counts the same
            # lines, so a restart sees the same spend.
            self._input_tokens += record["input_tokens"] or 0
            append_jsonl(self._log_path, record)
        # Raised here, not inside the except block above, so the SDK's error is not chained to
        # ours: it can quote the server's reply, which we only pass on with the key blanked out.
        if record["status"] == "error":
            raise BackendError(f"the Jev call for {request.file_id} failed: {record['error']}")
        choice = response.choices[GROUP_QUESTION]
        return SortResult(
            file_id=request.file_id,
            probs={group.id: choice.probabilities[group.name] for group in request.groups},
            fits_any=response.nouls[FITS_ANY_QUESTION].noul,
            model=response.model,
            option_order=option_order,
            input_tokens=record["input_tokens"],
            request_id=record["request_id"],
            latency_ms=record["latency_ms"],
        )

    def close(self) -> None:
        """Close the HTTP client. The backend cannot be used after this."""
        self._client.close()

    def _check_budget(self) -> None:
        """Raise BudgetExceeded once the spend so far has reached the cap. Needs the lock held."""
        spent = self._input_tokens / 1e6 * self._config.usd_per_million_tokens
        if spent >= self._config.max_spend_usd:
            raise BudgetExceeded(
                f"spend cap reached: ${spent:.4f} spent, the cap is "
                f"${self._config.max_spend_usd:g} (max_spend_usd in the vault config); "
                "nothing was sent"
            )

    def _problem(self, response: SystemOneResponse, groups: list[Group]) -> str | None:
        """Say why a response cannot be used as an answer, or return None when it can."""
        if response.model != self._config.model:
            return f"{response.model} answered, but {self._config.model} is the pinned model"
        choice = response.choices.get(GROUP_QUESTION)
        if choice is None:
            return f"the response has no choice answer named {GROUP_QUESTION!r}"
        if FITS_ANY_QUESTION not in response.nouls:
            return f"the response has no yes/no answer named {FITS_ANY_QUESTION!r}"
        missing = [group.name for group in groups if group.name not in choice.probabilities]
        if missing:
            return "the response has no probability for " + ", ".join(missing)
        return None

    def _without_key(self, text: str) -> str:
        """Blank out the API key in text from the SDK or the server, in case either quotes it."""
        return text.replace(self._api_key, "***")
