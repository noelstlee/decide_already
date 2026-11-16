"""The Jev backend against a fake TypeSafe endpoint. Nothing here touches the network."""

import dataclasses
import errno
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import httpx2
import pytest

from decide_already import backends
from decide_already.backends import jev, make_backend
from decide_already.backends.base import (
    BackendError,
    BudgetExceeded,
    LogFailed,
    RunStopped,
    SortRequest,
)
from decide_already.backends.jev import JevBackend
from decide_already.config import Config
from decide_already.log import read_jsonl, total_input_tokens
from decide_already.questions import build_questions
from decide_already.types import Group
from fakes import FakeBackend

KEY = "test-key-not-a-real-one"
PIN = "jev-1.14.0"
FILE_ID = "hw3/hw3_solution.py"
TOKENS = 2710

GROUPS = [
    Group(id="hw", name="HW", covers="homework specs, starter code", excludes="project files"),
    Group(id="logistics", name="Logistics", covers="syllabus, schedule, deadlines"),
    Group(id="project", name="Project", covers="project specs, examples", excludes="deadlines"),
    Group(id="lecture-notes", name="Lecture slides and notes", covers="lecture slides and notes"),
    Group(id="exams", name="Exams", covers="quizzes, midterms and finals"),
    Group(id="readings", name="Readings", covers="assigned papers and book chapters"),
]
ID_BY_NAME = {group.name: group.id for group in GROUPS}
PROBS_BY_NAME = {
    "HW": 0.61,
    "Logistics": 0.04,
    "Project": 0.2,
    "Lecture slides and notes": 0.08,
    "Exams": 0.04,
    "Readings": 0.03,
}
PROBS_BY_ID = {ID_BY_NAME[name]: prob for name, prob in PROBS_BY_NAME.items()}
STATE = {
    "file": {"name": "hw3_solution.py", "path": FILE_ID, "type": "Python code"},
    "title": "Homework 3: Kernel Methods",
    "outline": ["rbf_kernel()", "train_svm()", "predict()"],
    "opening": "import numpy as np\n\ndef rbf_kernel(a, b, gamma):\n    ...",
}
LOG_FIELDS = [
    "at",
    "file_id",
    "set_version",
    "variant",
    "backend",
    "model_requested",
    "model_answered",
    "option_order",
    "request",
    "response",
    "input_tokens",
    "request_id",
    "latency_ms",
    "status",
    "error",
]


def answer(model: str = PIN, tokens: int = TOKENS) -> dict:
    """A well-formed response body: a choice over the six groups and a yes/no."""
    return {
        "model": model,
        "answers": {
            "group": {
                "type": "choice",
                "choice": "HW",
                "confidence": 0.52,
                "probabilities": dict(PROBS_BY_NAME),
            },
            "fits_any": {"type": "noul", "noul": 0.94},
        },
        "usage": {"input_tokens": tokens, "output_tokens": 30},
    }


def unreadable_answer() -> dict:
    """A 200 body the pinned SDK cannot validate: the choice's confidence came back as a word."""
    body = answer()
    body["answers"]["group"]["confidence"] = "high"
    return body


def ok(body: dict | None = None, request_id: str = "req_test") -> httpx2.Response:
    """A 200 response carrying `body` (a normal answer by default) and a request id header."""
    return httpx2.Response(
        200, json=answer() if body is None else body, headers={"x-typesafe-request-id": request_id}
    )


def refusal(status: int) -> httpx2.Response:
    """An error response with the given status."""
    return httpx2.Response(status, json={"error": {"message": "the server said no"}})


class Server:
    """A fake TypeSafe endpoint: keeps every request and answers with `reply(request, number)`."""

    def __init__(self, reply=None):
        """Without `reply`, every request gets a normal answer. Requests are numbered from 1."""
        self.requests: list[httpx2.Request] = []
        self._reply = reply or (lambda request, number: ok())
        self._lock = threading.Lock()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Handle one request the way httpx2.MockTransport expects."""
        with self._lock:
            self.requests.append(request)
            number = len(self.requests)
        return self._reply(request, number)

    @property
    def bodies(self) -> list[dict]:
        """The JSON body of every request received, in order."""
        return [json.loads(request.content) for request in self.requests]


class Wire:
    """Where a client built without its own transport sends. It fails unless given a handler."""

    def __init__(self):
        """Start with no handler, so any request is a test failure."""
        self.requests: list[httpx2.Request] = []
        self.handler = None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Pass one request to the handler a test set, or fail the test."""
        if self.handler is None:
            raise AssertionError("a test tried to reach the network")
        self.requests.append(request)
        return self.handler(request)


@pytest.fixture(autouse=True)
def wire(monkeypatch):
    """Keep every test off the network, quick, and away from the real environment.

    Clients send bodies uncompressed, so handlers can read them, and back off for a millisecond.
    A client built without a transport (as make_backend does) sends to the returned Wire.
    """
    wire = Wire()
    real_client = jev.TypeSafeClient

    def client(**kwargs):
        """Build the real SDK client, but never with the real network."""
        if kwargs.get("transport") is None:
            kwargs["transport"] = httpx2.MockTransport(wire)
        return real_client(compression=False, **kwargs)

    quick = dataclasses.replace(jev.RETRY_POLICY, backoff_initial=0.001, backoff_max=0.001)
    monkeypatch.setattr(jev, "TypeSafeClient", client)
    monkeypatch.setattr(jev, "RETRY_POLICY", quick)
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return wire


def backend_on(
    server, tmp_path, *, config: Config | None = None, seed: int = 7, kind: type = JevBackend
) -> JevBackend:
    """A JevBackend (or its subclass `kind`) sending to `server`, logging to tmp_path/jev.jsonl."""
    return kind(
        config or Config(),
        tmp_path / "jev.jsonl",
        api_key=KEY,
        rng=random.Random(seed),
        transport=httpx2.MockTransport(server),
    )


def request_for(file_id: str = FILE_ID, **overrides) -> SortRequest:
    """A sort request for one file over the six groups, under set version 2."""
    fields = {"file_id": file_id, "state": STATE, "groups": GROUPS, "set_version": 2}
    return SortRequest(**{**fields, **overrides})


def sent_order(body: dict) -> list[str]:
    """The group ids in the order their options appear in a sent request body."""
    return [ID_BY_NAME[name] for name in body["questions"]["group"]["criteria"]]


# The request and the answer


def test_request_body_is_the_pin_the_state_and_the_questions_as_built(tmp_path):
    server = Server()
    backend_on(server, tmp_path, seed=7).sort(request_for())
    questions, _ = build_questions(GROUPS, random.Random(7))
    assert server.bodies == [{"model": PIN, "state": STATE, "questions": questions}]
    assert server.requests[0].url.path == "/v1/systemone"
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_the_model_sent_is_the_one_in_the_config(tmp_path):
    server = Server(lambda request, number: ok(answer(model="jev-1.13.0")))
    backend = backend_on(server, tmp_path, config=Config(model="jev-1.13.0"))
    for _ in range(3):
        assert backend.sort(request_for()).model == "jev-1.13.0"
    assert [body["model"] for body in server.bodies] == ["jev-1.13.0"] * 3


def test_sent_option_order_is_the_logged_and_the_returned_order(tmp_path):
    server = Server()
    result = backend_on(server, tmp_path).sort(request_for())
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    order = sent_order(server.bodies[0])
    assert order == line["option_order"] == result.option_order
    assert sorted(order) == sorted(ID_BY_NAME.values())
    assert list(server.bodies[0]["questions"]["fits_any"]["instructions"]["groups"]) == list(
        server.bodies[0]["questions"]["group"]["criteria"]
    )


def test_one_seed_is_reproducible_and_two_seeds_differ(tmp_path):
    orders = {}
    for label, seed in [("first", 1), ("again", 1), ("other", 2)]:
        (tmp_path / label).mkdir()
        backend = backend_on(Server(), tmp_path / label, seed=seed)
        orders[label] = [backend.sort(request_for()).option_order for _ in range(3)]
    assert orders["first"] == orders["again"]
    assert orders["first"] != orders["other"]


def test_every_call_shuffles_again(tmp_path):
    server = Server()
    backend = backend_on(server, tmp_path)
    for _ in range(10):
        backend.sort(request_for())
    assert len({tuple(sent_order(body)) for body in server.bodies}) > 1


def test_probabilities_keyed_by_name_come_back_keyed_by_group_id(tmp_path):
    result = backend_on(Server(), tmp_path).sort(request_for())
    assert result.probs == PROBS_BY_ID
    assert result.fits_any == 0.94
    assert result.file_id == FILE_ID
    assert result.model == PIN
    assert result.input_tokens == TOKENS
    assert result.request_id == "req_test"
    assert result.latency_ms >= 0


def test_names_only_sends_the_names_without_descriptions(tmp_path):
    server = Server()
    backend = backend_on(server, tmp_path)
    result = backend.sort(request_for(variant="names_only", with_descriptions=False))
    questions = server.bodies[0]["questions"]
    assert set(questions["group"]["criteria"]) == set(ID_BY_NAME)
    assert all(value is None for value in questions["group"]["criteria"].values())
    assert questions["fits_any"]["instructions"]["groups"] == list(questions["group"]["criteria"])
    assert all(group.covers not in json.dumps(questions) for group in GROUPS)
    assert result.probs == PROBS_BY_ID
    assert read_jsonl(tmp_path / "jev.jsonl")[0]["variant"] == "names_only"


# The log


def test_log_line_has_every_field_in_the_contract(tmp_path):
    server = Server()
    result = backend_on(server, tmp_path).sort(request_for())
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert list(line) == LOG_FIELDS
    at = datetime.fromisoformat(line["at"])
    assert at.utcoffset() == timedelta(0)
    assert abs(datetime.now(at.tzinfo) - at) < timedelta(minutes=1)
    assert line["file_id"] == FILE_ID
    assert line["set_version"] == 2
    assert line["variant"] == "card"
    assert line["backend"] == "jev"
    assert line["model_requested"] == line["model_answered"] == PIN
    assert line["option_order"] == result.option_order
    assert line["request"] == server.bodies[0]
    assert line["response"] == answer()
    assert line["input_tokens"] == TOKENS
    assert line["request_id"] == "req_test"
    assert line["latency_ms"] == result.latency_ms
    assert line["status"] == "ok"
    assert line["error"] is None


def test_a_response_without_a_request_id_is_still_an_answer(tmp_path):
    server = Server(lambda request, number: httpx2.Response(200, json=answer()))
    result = backend_on(server, tmp_path).sort(request_for())
    assert result.request_id is None
    assert read_jsonl(tmp_path / "jev.jsonl")[0]["request_id"] is None


# Failures: retried when that can help, logged, and never treated as an answer


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_a_retryable_status_followed_by_a_200_succeeds(tmp_path, status):
    server = Server(lambda request, number: refusal(status) if number == 1 else ok())
    result = backend_on(server, tmp_path).sort(request_for())
    assert result.probs == PROBS_BY_ID
    assert len(server.requests) == 2
    assert server.bodies[0] == server.bodies[1]
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["status"] == "ok"


def test_the_wait_before_a_retry_is_what_retry_after_asks(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(time, "sleep", waits.append)
    slow_down = {"error": {"message": "rate limited"}}
    server = Server(
        lambda request, number: (
            httpx2.Response(429, json=slow_down, headers={"retry-after": "2"})
            if number == 1
            else ok()
        )
    )
    assert backend_on(server, tmp_path).sort(request_for()).probs == PROBS_BY_ID
    assert waits == [2.0]


def test_a_persistent_500_raises_and_writes_an_error_line(tmp_path):
    server = Server(lambda request, number: refusal(500))
    backend = backend_on(server, tmp_path)
    with pytest.raises(BackendError, match="500") as caught:
        backend.sort(request_for())
    assert not isinstance(caught.value, BudgetExceeded)
    assert FILE_ID in str(caught.value)
    assert len(server.requests) == jev.RETRY_POLICY.max_retries + 1
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert list(line) == LOG_FIELDS
    assert line["status"] == "error"
    assert line["response"] is None
    assert "500" in line["error"]
    assert "the server said no" in line["error"]
    assert line["model_requested"] == PIN
    assert line["model_answered"] is None
    assert line["input_tokens"] is None
    assert line["request"] == server.bodies[0]
    assert line["option_order"] == sent_order(server.bodies[0])


def test_a_bad_request_is_not_retried(tmp_path):
    server = Server(lambda request, number: refusal(400))
    with pytest.raises(BackendError, match="400"):
        backend_on(server, tmp_path).sort(request_for())
    assert len(server.requests) == 1
    assert read_jsonl(tmp_path / "jev.jsonl")[0]["status"] == "error"


def test_a_connection_failure_raises_and_is_logged(tmp_path):
    def unreachable(request, number):
        """Fail the way a refused connection does."""
        raise httpx2.ConnectError("connection refused", request=request)

    server = Server(unreachable)
    with pytest.raises(BackendError, match="connection refused"):
        backend_on(server, tmp_path).sort(request_for())
    assert len(server.requests) == jev.RETRY_POLICY.max_retries + 1
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["status"] == "error"
    assert line["response"] is None
    assert line["request_id"] is None


def test_an_answering_model_other_than_the_pin_raises(tmp_path):
    server = Server(lambda request, number: ok(answer(model="jev-1.13.0")))
    with pytest.raises(BackendError, match="jev-1.13.0 answered, but jev-1.14.0 is the pinned"):
        backend_on(server, tmp_path).sort(request_for())
    assert len(server.requests) == 1
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["status"] == "error"
    assert line["model_requested"] == PIN
    assert line["model_answered"] == "jev-1.13.0"
    assert line["response"] == answer(model="jev-1.13.0")
    assert "jev-1.13.0" in line["error"]


@pytest.mark.parametrize("question", ["group", "fits_any"])
def test_a_response_missing_an_answer_raises(tmp_path, question):
    incomplete = answer()
    del incomplete["answers"][question]
    server = Server(lambda request, number: ok(incomplete))
    with pytest.raises(BackendError, match=question):
        backend_on(server, tmp_path).sort(request_for())
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["status"] == "error"
    assert line["response"] == incomplete


def test_a_response_missing_a_groups_probability_raises(tmp_path):
    partial = answer()
    partial["answers"]["group"]["probabilities"] = {"HW": 0.9, "Logistics": 0.1}
    server = Server(lambda request, number: ok(partial))
    with pytest.raises(BackendError, match="no probability for Project"):
        backend_on(server, tmp_path).sort(request_for())
    assert read_jsonl(tmp_path / "jev.jsonl")[0]["status"] == "error"


@pytest.mark.parametrize(
    "body", [{"text": "<html>proxy</html>"}, {"json": ["not", "an", "answer"]}]
)
def test_a_body_that_is_not_an_answer_raises(tmp_path, body):
    """A 200 whose body is not a JSON object is no answer, so its line keeps no response."""
    server = Server(lambda request, number: httpx2.Response(200, **body))
    with pytest.raises(BackendError):
        backend_on(server, tmp_path).sort(request_for())
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["status"] == "error"
    assert line["response"] is None
    assert line["input_tokens"] is None


# An answer the SDK cannot validate: answered and billed, so kept and counted


def test_an_answer_the_sdk_cannot_validate_keeps_its_body_tokens_and_ids(tmp_path):
    """The call fails, but its line holds what came back: the body, the tokens, the ids."""
    drifted = unreadable_answer()
    server = Server(lambda request, number: ok(drifted))
    with pytest.raises(BackendError, match="Invalid response data") as caught:
        backend_on(server, tmp_path).sort(request_for())
    assert not isinstance(caught.value, RunStopped)
    assert len(server.requests) == 1
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert list(line) == LOG_FIELDS
    assert line["status"] == "error"
    assert "answers.group.confidence" in line["error"]
    assert line["response"] == drifted
    assert line["input_tokens"] == TOKENS
    assert line["request_id"] == "req_test"
    assert line["model_requested"] == line["model_answered"] == PIN


def test_answers_the_sdk_cannot_validate_count_toward_the_spend_cap(tmp_path):
    """Two of them reach a cap of two calls, for this backend and for one started later."""
    config = Config(usd_per_million_tokens=1.0, max_spend_usd=2 * TOKENS / 1e6)
    server = Server(lambda request, number: ok(unreadable_answer()))
    backend = backend_on(server, tmp_path, config=config)
    for _ in range(2):
        with pytest.raises(BackendError, match="Invalid response data"):
            backend.sort(request_for())
    assert total_input_tokens(tmp_path / "jev.jsonl") == 2 * TOKENS
    for same_log in (backend, backend_on(server, tmp_path, config=config)):
        with pytest.raises(BudgetExceeded):
            same_log.sort(request_for())
    assert len(server.requests) == 2


@pytest.mark.parametrize(
    "usage", [None, {"input_tokens": "2710"}, {"input_tokens": True}, {"input_tokens": 2.5}, [2710]]
)
def test_a_token_count_that_cannot_be_read_is_not_logged_as_one(tmp_path, usage):
    """Spend is summed from the log, so only a whole number may be logged as input_tokens."""
    body = answer()
    if usage is None:
        del body["usage"]
    else:
        body["usage"] = usage
    server = Server(lambda request, number: ok(body))
    with pytest.raises(BackendError, match="Invalid response data"):
        backend_on(server, tmp_path).sort(request_for())
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["response"] == body
    assert line["input_tokens"] is None
    assert total_input_tokens(tmp_path / "jev.jsonl") == 0


def test_an_unvalidated_answer_without_a_model_name_logs_none(tmp_path):
    """model_answered is only what the body says: nothing is filled in for it."""
    body = answer()
    del body["model"]
    server = Server(lambda request, number: ok(body))
    with pytest.raises(BackendError, match="Invalid response data"):
        backend_on(server, tmp_path).sort(request_for())
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["model_answered"] is None
    assert line["response"] == body
    assert line["input_tokens"] == TOKENS


# A call that is interrupted: logged before the interruption goes on


@pytest.mark.parametrize(
    ("interruption", "logged_as"),
    [
        (KeyboardInterrupt(), "interrupted: KeyboardInterrupt"),
        (RuntimeError("the socket broke"), "interrupted: RuntimeError: the socket broke"),
    ],
    ids=["ctrl_c", "not_an_sdk_error"],
)
def test_an_interrupted_call_leaves_a_line_and_the_interruption_goes_on(
    tmp_path, interruption, logged_as
):
    """Ctrl+C, or any error that is not the SDK's own, while the request is out."""

    def cut_off(request, number):
        """Break off the request the way Ctrl+C or a failing layer under the SDK does."""
        raise interruption

    server = Server(cut_off)
    with pytest.raises(type(interruption)) as caught:
        backend_on(server, tmp_path).sort(request_for())
    assert caught.value is interruption
    assert len(server.requests) == 1
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert list(line) == LOG_FIELDS
    assert line["status"] == "error"
    assert line["error"] == logged_as
    assert line["response"] is None
    assert line["input_tokens"] is None
    assert line["request"] == server.bodies[0]
    assert line["option_order"] == sent_order(server.bodies[0])
    assert line["latency_ms"] >= 0


def test_an_interrupt_after_the_answer_was_read_keeps_the_answer_in_its_line(tmp_path, monkeypatch):
    """The line then holds the response and its tokens, and the call counts as spent."""

    def ctrl_c(self, response, groups):
        """Stand in for the check of the answer, with Ctrl+C arriving during it."""
        raise KeyboardInterrupt

    config = Config(usd_per_million_tokens=1.0, max_spend_usd=TOKENS / 1e6)
    server = Server()
    backend = backend_on(server, tmp_path, config=config)
    with monkeypatch.context() as patch:
        patch.setattr(JevBackend, "_problem", ctrl_c)
        with pytest.raises(KeyboardInterrupt):
            backend.sort(request_for())
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["status"] == "error"
    assert line["error"] == "interrupted: KeyboardInterrupt"
    assert line["response"] == answer()
    assert line["input_tokens"] == TOKENS
    assert line["request_id"] == "req_test"
    with pytest.raises(BudgetExceeded):
        backend.sort(request_for())
    assert len(server.requests) == 1


# A log that cannot be written: nothing is sent, and the run stops


def folder_is_a_file(log) -> None:
    """Put a plain file where the log's folder should be."""
    log.parent.write_text("not a folder")


def log_is_a_folder(log) -> None:
    """Put a folder where the log file should be."""
    log.mkdir(parents=True)


def log_is_read_only(log) -> None:
    """Leave an empty log that may only be read, as a copy from a read-only place can be."""
    log.parent.mkdir()
    log.touch()
    log.chmod(0o444)
    if os.access(log, os.W_OK):
        pytest.skip("this user may write to a read-only file")


@pytest.mark.parametrize("spoil", [folder_is_a_file, log_is_a_folder, log_is_read_only])
def test_a_log_that_cannot_be_opened_stops_the_call_before_it_is_sent(tmp_path, spoil):
    """Every call must be logged, so with no way to log it nothing goes out."""
    log = tmp_path / ".decide" / "jev.jsonl"
    server = Server()
    backend = backend_on(server, log.parent)
    spoil(log)
    for _ in range(2):
        with pytest.raises(LogFailed, match="nothing was sent") as caught:
            backend.sort(request_for())
        assert isinstance(caught.value, RunStopped)
        assert isinstance(caught.value.__cause__, OSError)
        assert str(log) in str(caught.value)
        assert caught.value.__cause__.strerror in str(caught.value)
    assert server.requests == []


def test_the_log_check_makes_the_folder_and_adds_no_line_of_its_own(tmp_path):
    """A first call in a vault with no .decide folder yet is sent, and logged as one line."""
    log = tmp_path / "new" / ".decide" / "jev.jsonl"
    server = Server()
    backend_on(server, log.parent).sort(request_for())
    assert len(server.requests) == 1
    assert [line["file_id"] for line in read_jsonl(log)] == [FILE_ID]


def full_disk(path, record) -> None:
    """Stand in for append_jsonl on a disk that filled up after the log was opened."""
    raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def test_an_answer_that_cannot_be_logged_stops_the_run(tmp_path, monkeypatch):
    """The call that went out is reported as unlogged, and no later call is sent."""
    log = tmp_path / "jev.jsonl"
    server = Server()
    backend = backend_on(server, tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(jev, "append_jsonl", full_disk)
        with pytest.raises(LogFailed, match=f"the call for {FILE_ID} is not in the log") as caught:
            backend.sort(request_for())
    assert isinstance(caught.value.__cause__, OSError)
    assert str(log) in str(caught.value)
    assert os.strerror(errno.ENOSPC) in str(caught.value)
    # From here the disk has room again and a line could be written. The backend still refuses.
    with pytest.raises(LogFailed, match="nothing was sent") as later:
        backend.sort(request_for("syllabus.pdf"))
    assert os.strerror(errno.ENOSPC) in str(later.value)
    assert len(server.requests) == 1
    assert read_jsonl(log) == []


def test_an_answer_that_could_not_be_logged_still_counts_as_spent(tmp_path, monkeypatch):
    """The log misses the call, but this backend's cap does not: one call was the cap."""
    monkeypatch.setattr(jev, "append_jsonl", full_disk)
    config = Config(usd_per_million_tokens=1.0, max_spend_usd=TOKENS / 1e6)
    server = Server()
    backend = backend_on(server, tmp_path, config=config)
    with pytest.raises(LogFailed, match="is not in the log"):
        backend.sort(request_for())
    with pytest.raises(BudgetExceeded):
        backend.sort(request_for())
    assert len(server.requests) == 1


def test_a_failed_call_that_cannot_be_logged_is_a_log_failure(tmp_path, monkeypatch):
    """A call the server refused went out too, so an unwritten line for it stops the run."""
    monkeypatch.setattr(jev, "append_jsonl", full_disk)
    server = Server(lambda request, number: refusal(400))
    with pytest.raises(LogFailed, match="is not in the log"):
        backend_on(server, tmp_path).sort(request_for())
    assert len(server.requests) == 1


def test_a_record_that_is_not_json_is_a_log_failure(tmp_path):
    """The SDK can send values a JSON line cannot hold, such as a date."""
    server = Server()
    backend = backend_on(server, tmp_path)
    with pytest.raises(LogFailed, match="not JSON serializable") as caught:
        backend.sort(request_for(state={**STATE, "seen": datetime(2026, 11, 15)}))
    assert isinstance(caught.value.__cause__, TypeError)
    assert len(server.requests) == 1
    assert read_jsonl(tmp_path / "jev.jsonl") == []


def test_an_interruption_goes_on_even_when_its_line_cannot_be_written(tmp_path, monkeypatch):
    """Ctrl+C is never turned into a log failure. The failure is attached to it as a note."""
    monkeypatch.setattr(jev, "append_jsonl", full_disk)

    def ctrl_c(request, number):
        """Break off the first request the way Ctrl+C does. A later one would be answered."""
        if number == 1:
            raise KeyboardInterrupt
        return ok()

    server = Server(ctrl_c)
    backend = backend_on(server, tmp_path)
    with pytest.raises(KeyboardInterrupt) as caught:
        backend.sort(request_for())
    (note,) = caught.value.__notes__
    assert str(tmp_path / "jev.jsonl") in note
    assert f"the call for {FILE_ID} is not in the log" in note
    with pytest.raises(LogFailed, match="nothing was sent"):
        backend.sort(request_for())
    assert len(server.requests) == 1


# The spend cap


def test_the_spend_cap_raises_and_sends_nothing(tmp_path):
    config = Config(max_spend_usd=0.0001)  # about 2,400 tokens at the default price
    server = Server()
    backend = backend_on(server, tmp_path, config=config)
    backend.sort(request_for())
    with pytest.raises(BudgetExceeded, match="spend cap reached"):
        backend.sort(request_for("syllabus.pdf"))
    assert len(server.requests) == 1
    assert [line["file_id"] for line in read_jsonl(tmp_path / "jev.jsonl")] == [FILE_ID]


def test_spend_is_counted_from_the_log_at_start(tmp_path):
    config = Config(max_spend_usd=0.0001)
    backend_on(Server(), tmp_path, config=config).sort(request_for())
    server = Server()
    with pytest.raises(BudgetExceeded):
        backend_on(server, tmp_path, config=config).sort(request_for())
    assert server.requests == []


def test_exactly_at_the_cap_counts_as_reached(tmp_path):
    config = Config(usd_per_million_tokens=1.0, max_spend_usd=2 * TOKENS / 1e6)
    server = Server()
    backend = backend_on(server, tmp_path, config=config)
    backend.sort(request_for())
    backend.sort(request_for())
    with pytest.raises(BudgetExceeded):
        backend.sort(request_for())
    assert len(server.requests) == 2


def test_failed_calls_without_an_answer_cost_nothing(tmp_path):
    config = Config(max_spend_usd=0.0001)
    server = Server(lambda request, number: refusal(400) if number <= 3 else ok())
    backend = backend_on(server, tmp_path, config=config)
    for _ in range(3):
        with pytest.raises(BackendError):
            backend.sort(request_for())
    assert backend.sort(request_for()).probs == PROBS_BY_ID
    assert total_input_tokens(tmp_path / "jev.jsonl") == TOKENS


def test_the_running_spend_matches_what_a_restart_reads_from_the_log(tmp_path):
    # The answer from another model version is rejected but was still billed, so it counts.
    config = Config(usd_per_million_tokens=1.0, max_spend_usd=4 * TOKENS / 1e6)
    replies = [ok, lambda: refusal(400), ok, lambda: ok(answer(model="jev-1.13.0")), ok]
    server = Server(lambda request, number: replies[number - 1]())
    backend = backend_on(server, tmp_path, config=config)
    outcomes = []
    for _ in replies:
        try:
            backend.sort(request_for())
            outcomes.append("ok")
        except BackendError:
            outcomes.append("error")
    assert outcomes == ["ok", "error", "ok", "error", "ok"]
    assert total_input_tokens(tmp_path / "jev.jsonl") == 4 * TOKENS
    for same_log in (backend, backend_on(server, tmp_path, config=config)):
        with pytest.raises(BudgetExceeded):
            same_log.sort(request_for())
    assert len(server.requests) == len(replies)


# The API key


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_a_missing_key_raises(tmp_path, missing):
    with pytest.raises(BackendError) as caught:
        JevBackend(Config(), tmp_path / "jev.jsonl", api_key=missing)
    assert "TYPESAFE_API_KEY" in str(caught.value)
    assert ".env" in str(caught.value)


def test_a_key_the_sdk_rejects_raises_without_quoting_it(tmp_path):
    with pytest.raises(BackendError) as caught:
        JevBackend(Config(), tmp_path / "jev.jsonl", api_key="two words")
    assert "two words" not in str(caught.value)


def test_the_key_never_appears_in_the_log_or_in_an_error(tmp_path):
    def reply(request, number):
        """Answer once, refuse once, then send back the Authorization header in an error."""
        if number == 1:
            return ok()
        if number == 2:
            return refusal(400)
        return httpx2.Response(401, json={"error": f"bad key: {request.headers['authorization']}"})

    server = Server(reply)
    backend = backend_on(server, tmp_path)
    backend.sort(request_for())
    with pytest.raises(BackendError) as refused:
        backend.sort(request_for())
    with pytest.raises(BackendError) as echoed:
        backend.sort(request_for())
    assert all(request.headers["authorization"] == f"Bearer {KEY}" for request in server.requests)
    assert len(read_jsonl(tmp_path / "jev.jsonl")) == 3
    assert KEY not in (tmp_path / "jev.jsonl").read_text()
    for caught in (refused, echoed):
        assert KEY not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    assert "bad key: Bearer ***" in str(echoed.value)


@pytest.mark.parametrize("readable", [True, False], ids=["an_answer", "an_unvalidated_answer"])
def test_a_response_body_that_quotes_the_key_is_logged_with_the_key_blanked_out(tmp_path, readable):
    """Whether or not the SDK can validate the body, it reaches the log without the key."""

    def reply(request, number):
        """Answer, quoting the Authorization header in a field the SDK does not know."""
        body = answer() if readable else unreadable_answer()
        body["debug"] = {"headers": [request.headers["authorization"]], f"key {KEY}": True}
        return ok(body)

    backend = backend_on(Server(reply), tmp_path)
    if readable:
        assert backend.sort(request_for()).probs == PROBS_BY_ID
    else:
        with pytest.raises(BackendError) as caught:
            backend.sort(request_for())
        assert KEY not in str(caught.value)
    assert KEY not in (tmp_path / "jev.jsonl").read_text()
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["response"]["debug"] == {"headers": ["Bearer ***"], "key ***": True}
    assert line["response"]["usage"] == answer()["usage"]


def test_an_interruption_that_quotes_the_key_is_logged_with_the_key_blanked_out(tmp_path):
    """The text of a foreign error goes into the log line, so the key is blanked in it too."""

    def cut_off(request, number):
        """Fail below the SDK with a message that quotes the Authorization header."""
        raise RuntimeError(f"cannot send {request.headers['authorization']}")

    with pytest.raises(RuntimeError):
        backend_on(Server(cut_off), tmp_path).sort(request_for())
    assert KEY not in (tmp_path / "jev.jsonl").read_text()
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["error"] == "interrupted: RuntimeError: cannot send Bearer ***"


def test_make_backend_without_a_key_raises(tmp_path, monkeypatch):
    """With no key to be found, the backend cannot be built.

    The key lookup is replaced: the real one also reads any .env file above the temp folder.
    """
    asked = []

    def no_key(name, start=None):
        """Stand in for config.env_value: remember what was asked for and find nothing (None)."""
        asked.append((name, start))

    monkeypatch.setattr(backends, "env_value", no_key)
    with pytest.raises(BackendError, match="TYPESAFE_API_KEY"):
        make_backend(tmp_path, Config())
    assert asked == [("TYPESAFE_API_KEY", tmp_path), ("TYPESAFE_API_KEY", None)]


def test_make_backend_uses_the_key_from_the_environment_and_the_vault_log(
    tmp_path, monkeypatch, wire
):
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    wire.handler = lambda request: ok()
    backend = make_backend(tmp_path, Config())
    assert backend.name == "jev"
    assert backend.sort(request_for()).probs == PROBS_BY_ID
    assert wire.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert [line["file_id"] for line in read_jsonl(tmp_path / ".decide" / "jev.jsonl")] == [FILE_ID]


def test_make_backend_finds_the_key_in_a_dotenv_file_above_the_vault(tmp_path, wire):
    (tmp_path / ".env").write_text(f"TYPESAFE_API_KEY={KEY}\n")
    root = tmp_path / "vault"
    root.mkdir()
    wire.handler = lambda request: ok()
    make_backend(root, Config()).sort(request_for())
    assert wire.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in (root / ".decide" / "jev.jsonl").read_text()


def test_make_backend_finds_the_key_in_the_folder_p0_runs_from(tmp_path, monkeypatch, wire):
    """A vault kept outside the repo: the key in the repo's .env is still found."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    repo, root = tmp_path / "repo", tmp_path / "elsewhere" / "vault"
    repo.mkdir()
    root.mkdir(parents=True)
    (repo / ".env").write_text(f"TYPESAFE_API_KEY={KEY}\n")
    monkeypatch.chdir(repo)
    wire.handler = lambda request: ok()
    make_backend(root, Config()).sort(request_for())
    assert wire.requests[0].headers["authorization"] == f"Bearer {KEY}"


# Threads


class OwnedLock:
    """A lock that knows which thread holds it, so a test can ask whether the caller does."""

    def __init__(self):
        """Start free."""
        self._lock = threading.Lock()
        self._owner = None

    def __enter__(self):
        """Take the lock and note the calling thread as its holder."""
        self._lock.acquire()
        self._owner = threading.get_ident()

    def __exit__(self, *exc_info):
        """Forget the holder and free the lock."""
        self._owner = None
        self._lock.release()

    def held_here(self) -> bool:
        """Say whether the calling thread holds the lock."""
        return self._owner == threading.get_ident()


class WatchedBackend(JevBackend):
    """A JevBackend that notes, at each change of its spend counter, who holds its lock."""

    def __setattr__(self, name, value):
        """Note whether the caller holds the lock when the counter changes after its first value."""
        if name == "_input_tokens" and name in vars(self):
            self.counter_changes.append(self._lock.held_here())
        super().__setattr__(name, value)


def test_threads_each_write_one_whole_line_and_the_spend_adds_up(tmp_path, monkeypatch):
    """Forty calls from eight threads: whole lines, an exact spend, and the lock behind both.

    The outcome alone can come out right without the lock, so the mechanism is checked too: the
    calling thread holds the backend's lock at every log append and every change of the counter.
    """
    calls = 40
    config = Config(usd_per_million_tokens=1.0, max_spend_usd=calls * TOKENS / 1e6)
    server = Server()
    backend = backend_on(server, tmp_path, config=config, kind=WatchedBackend)
    backend._lock = OwnedLock()
    backend.counter_changes = []
    appends = []
    real_append = jev.append_jsonl

    def watched_append(path, record):
        """Note whether the caller holds the lock, then append as usual."""
        appends.append(backend._lock.held_here())
        real_append(path, record)

    monkeypatch.setattr(jev, "append_jsonl", watched_append)
    long_state = {**STATE, "opening": "kernel methods " * 2000}
    file_ids = [f"notes/file_{number}.md" for number in range(calls)]

    def sort(file_id):
        """Sort one file with a state long enough that its log line is far over 8 KB."""
        return backend.sort(request_for(file_id, state=long_state))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(sort, file_ids))

    assert appends == [True] * calls
    assert backend.counter_changes == [True] * calls
    assert [result.file_id for result in results] == file_ids
    lines = (tmp_path / "jev.jsonl").read_text().splitlines()
    assert len(lines) == calls
    records = [json.loads(line) for line in lines]
    assert sorted(record["file_id"] for record in records) == sorted(file_ids)
    for record in records:
        assert list(record) == LOG_FIELDS
        assert record["status"] == "ok"
        assert record["option_order"] == sent_order(record["request"])
    assert total_input_tokens(tmp_path / "jev.jsonl") == calls * TOKENS
    with pytest.raises(BudgetExceeded):
        backend.sort(request_for("one_more.md"))
    assert len(server.requests) == calls


# Closing


def test_close_closes_the_transport(tmp_path):
    class Transport(httpx2.MockTransport):
        """A mock transport that remembers being closed."""

        closed = False

        def close(self):
            """Note the close instead of releasing anything."""
            self.closed = True

    transport = Transport(Server())
    backend = JevBackend(Config(), tmp_path / "jev.jsonl", api_key=KEY, transport=transport)
    backend.close()
    assert transport.closed


# The fake backend other tests use


def test_fake_backend_answers_from_its_table():
    fake = FakeBackend(answers={FILE_ID: ({"hw": 0.8, "logistics": 0.2}, 0.95)})
    result = fake.sort(request_for())
    assert result.probs == {"hw": 0.8, "logistics": 0.2}
    assert result.fits_any == 0.95
    assert result.file_id == FILE_ID
    assert result.model == "fake-0"
    assert result.option_order == [group.id for group in GROUPS]
    assert fake.name == "fake"


def test_fake_backend_uses_the_default_for_other_files():
    fake = FakeBackend(
        answers={FILE_ID: ({"hw": 1.0}, 0.9)}, default=({"logistics": 0.7, "hw": 0.3}, 0.4)
    )
    assert fake.sort(request_for("syllabus.pdf")).probs == {"logistics": 0.7, "hw": 0.3}
    assert fake.sort(request_for("syllabus.pdf")).fits_any == 0.4
    assert fake.sort(request_for()).probs == {"hw": 1.0}


def test_fake_backend_spreads_evenly_without_an_answer_or_a_default():
    result = FakeBackend().sort(request_for())
    assert set(result.probs) == set(ID_BY_NAME.values())
    assert all(prob == pytest.approx(1 / len(GROUPS)) for prob in result.probs.values())
    assert sum(result.probs.values()) == pytest.approx(1.0)
    assert result.fits_any == 0.9


def test_fake_backend_fails_for_the_listed_files_only():
    fake = FakeBackend(fail={"broken.pdf"})
    with pytest.raises(BackendError):
        fake.sort(request_for("broken.pdf"))
    assert fake.sort(request_for()).file_id == FILE_ID


def test_fake_backend_remembers_every_request_in_order():
    fake = FakeBackend(fail={"broken.pdf"})
    first = request_for("a.md")
    second = request_for("broken.pdf")
    fake.sort(first)
    with pytest.raises(BackendError):
        fake.sort(second)
    assert fake.calls == [first, second]


def test_fake_backend_hands_out_copies_of_its_answers():
    fake = FakeBackend(default=({"hw": 1.0}, 0.9))
    fake.sort(request_for()).probs["hw"] = 0.0
    assert fake.sort(request_for()).probs == {"hw": 1.0}
