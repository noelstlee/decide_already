"""The Jev backend against a fake TypeSafe endpoint. Nothing here touches the network."""

import dataclasses
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import httpx2
import pytest

from decide_already.backends import jev, make_backend
from decide_already.backends.base import BackendError, BudgetExceeded, SortRequest
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


def backend_on(server, tmp_path, *, config: Config | None = None, seed: int = 7) -> JevBackend:
    """A JevBackend that sends to `server` and logs to tmp_path/jev.jsonl."""
    return JevBackend(
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


def test_a_body_that_is_not_an_answer_raises(tmp_path):
    server = Server(lambda request, number: httpx2.Response(200, text="<html>proxy</html>"))
    with pytest.raises(BackendError):
        backend_on(server, tmp_path).sort(request_for())
    (line,) = read_jsonl(tmp_path / "jev.jsonl")
    assert line["status"] == "error"
    assert line["response"] is None


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


def test_make_backend_without_a_key_raises(tmp_path):
    with pytest.raises(BackendError, match="TYPESAFE_API_KEY"):
        make_backend(tmp_path, Config())


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


# Threads


def test_threads_each_write_one_whole_line_and_the_spend_adds_up(tmp_path):
    calls = 40
    config = Config(usd_per_million_tokens=1.0, max_spend_usd=calls * TOKENS / 1e6)
    server = Server()
    backend = backend_on(server, tmp_path, config=config)
    long_state = {**STATE, "opening": "kernel methods " * 2000}
    file_ids = [f"notes/file_{number}.md" for number in range(calls)]

    def sort(file_id):
        """Sort one file with a state long enough that its log line is far over 8 KB."""
        return backend.sort(request_for(file_id, state=long_state))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(sort, file_ids))

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
