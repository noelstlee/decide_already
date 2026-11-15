"""Tests for the three evaluation baselines. Nothing here touches the network or a real key."""

import json
import math
import sys
from types import SimpleNamespace

import anthropic
import httpx2
import numpy as np
import pytest

from decide_already.backends.base import BackendError, SortRequest, state_text
from decide_already.config import Config
from decide_already.eval.baselines import embed as embed_module
from decide_already.eval.baselines import llm
from decide_already.eval.baselines.embed import (
    DEFAULT_MODEL_NAME,
    EmbeddingBackend,
    chunks,
    cosine,
    group_text,
    mean_vector,
    softmax,
)
from decide_already.eval.baselines.rules import DEFAULT_RULES, Rule, RulesBackend, path_words
from decide_already.eval.groups import EVAL_GROUPS
from decide_already.eval.metrics import Cutoffs, Prediction, outcome
from decide_already.questions import GROUP_INSTRUCTIONS
from decide_already.types import Group

EVAL_IDS = [group.id for group in EVAL_GROUPS]


def ask(backend, file_id, groups=EVAL_GROUPS, state=None, **extra):
    """Send one file to a backend. By default the state holds only the file's name and path."""
    if state is None:
        state = {"file": {"name": file_id.rsplit("/", 1)[-1], "path": file_id, "type": "PDF"}}
    request = SortRequest(file_id=file_id, state=state, groups=groups, set_version=1, **extra)
    return backend.sort(request)


def top(result):
    """The group id with the highest probability in a result."""
    return max(result.probs, key=result.probs.get)


# Rules baseline


@pytest.mark.parametrize(
    ("file_id", "group_id"),
    [
        ("syllabus.pdf", "logistics"),
        ("exam_schedule.pdf", "logistics"),  # logistics words are checked first
        ("final_project_guidelines.pdf", "projects"),  # projects before exams ("final")
        ("midterm_solutions.pdf", "exams"),  # exams before homework ("solutions")
        ("quiz2.pdf", "exams"),
        ("hw3/hw3_solution.py", "homework"),
        ("ps4.pdf", "homework"),
        ("ProblemSet2.pdf", "homework"),  # camelCase splits into problem and set
        ("lab1/notes.txt", "homework"),  # "lab" beats the weaker "notes"
        ("lectures/lec05_kernels.pdf", "lecture-notes"),
        ("lecture3_exercises.pdf", "lecture-notes"),  # "lecture" beats the weaker "exercises"
        ("readings/bishop_ch3.pdf", "lecture-notes"),
        ("course_info.html", "logistics"),
        ("utils.py", "homework"),  # no telling word: the extension decides
        ("week2.pptx", "lecture-notes"),
    ],
)
def test_default_rules_follow_their_priority_order(file_id, group_id):
    """The first rule that matches wins, and it gets the whole probability."""
    result = ask(RulesBackend(), file_id)
    assert result.probs == {gid: 1.0 if gid == group_id else 0.0 for gid in EVAL_IDS}
    assert result.fits_any == 1.0


def test_default_rules_cover_exactly_the_evaluation_groups():
    """Every default rule points at an evaluation group, and every group has a rule."""
    assert {rule.group_id for rule in DEFAULT_RULES} == set(EVAL_IDS)


def test_rules_no_match_makes_no_pick():
    """A name with no keyword and no known extension matches nothing: no pick at all."""
    result = ask(RulesBackend(), "kernel_methods_2006.pdf")
    assert result.fits_any == 0.0
    assert result.probs == {}


def test_rules_result_names_the_backend_and_the_option_order():
    """The result carries the fixed model name, the groups in request order, and no tokens."""
    backend = RulesBackend(DEFAULT_RULES)
    result = ask(backend, "syllabus.pdf")
    assert backend.name == "rules"
    assert result.model == "rules"
    assert result.option_order == EVAL_IDS
    assert result.input_tokens is None
    assert result.latency_ms >= 0


def test_rules_first_match_wins_in_the_order_given():
    """The same two rules in the other order give the other group."""
    first_b = [Rule(group_id="b", keywords=["x"]), Rule(group_id="a", keywords=["x"])]
    groups = [Group(id="a", name="A", covers="a"), Group(id="b", name="B", covers="b")]
    assert top(ask(RulesBackend(first_b), "x.pdf", groups)) == "b"
    assert top(ask(RulesBackend(first_b[::-1]), "x.pdf", groups)) == "a"


def test_rules_keywords_match_whole_words_only():
    """A keyword inside a longer word is not a match: no lab in syllabus, no exam in examples."""
    labs = RulesBackend([Rule(group_id="homework", keywords=["lab"])])
    assert ask(labs, "syllabus.pdf").fits_any == 0.0
    assert top(ask(labs, "Lab_3.pdf")) == "homework"
    assert top(ask(RulesBackend(), "examples.py")) == "homework"  # by extension, not "exam"


def test_rules_extension_is_not_a_word():
    """A PostScript file is not homework just because its extension spells "ps"."""
    assert path_words("hw3/ProblemSet2_soln.pdf") == {"hw", "problem", "set", "soln"}
    assert top(ask(RulesBackend(), "notes.ps")) == "lecture-notes"


def test_rules_extension_match_ignores_case():
    """An upper-case extension matches its lower-case rule."""
    assert top(ask(RulesBackend(), "DEMO.PY")) == "homework"


def test_rules_skip_groups_that_are_not_offered():
    """With no exams group, the exams rule cannot fire: the next matching rule decides."""
    without_exams = [group for group in EVAL_GROUPS if group.id != "exams"]
    assert ask(RulesBackend(), "midterm.pdf", without_exams).fits_any == 0.0
    result = ask(RulesBackend(), "midterm_solutions.pdf", without_exams)
    assert top(result) == "homework"
    assert "exams" not in result.probs


def test_rules_never_read_the_content():
    """Words in the card's text do not count. Only the name and path do."""
    state = {
        "file": {"name": "scan0001.pdf", "path": "scan0001.pdf", "type": "PDF"},
        "title": "Syllabus",
        "opening": "Homework 3 is due before the midterm exam. See the lecture notes.",
    }
    assert ask(RulesBackend(), "scan0001.pdf", state=state).fits_any == 0.0


def test_rules_fall_back_to_the_file_id_when_the_card_has_no_name():
    """The no-name card variant drops name and path, but the request still has the file id."""
    result = ask(RulesBackend(), "hw1/spec.pdf", state={"file": {"type": "PDF, 2 pages"}})
    assert result.fits_any == 1.0
    assert top(result) == "homework"
    # The extension comes from the file id as well.
    by_extension = ask(RulesBackend(), "tools/run.sh", state={"file": {"type": "Shell script"}})
    assert top(by_extension) == "homework"


# Embedding baseline

VOCAB = ("homework", "lecture", "exam")


def count_words(text):
    """A toy embedding: how often each of three words appears in the text."""
    return [float(text.lower().count(word)) for word in VOCAB]


class FakeEmbed:
    """Stands in for the embedding model, and records every batch it is asked to embed."""

    def __init__(self):
        """Start with no recorded calls."""
        self.calls = []

    def __call__(self, texts):
        """Embed each text with count_words."""
        self.calls.append(list(texts))
        return [count_words(text) for text in texts]


SMALL_GROUPS = [
    Group(id="hw", name="Homework", covers="homework problems"),
    Group(id="lec", name="Lecture notes", covers="lecture slides"),
    Group(id="ex", name="Exams", covers="exam papers", excludes="homework"),
]
# With descriptions and excludes the groups embed as [2, 0, 0], [0, 2, 0] and [1, 0, 2].
# Without excludes (the default) the last one is [0, 0, 2].
# This card embeds as [1, 2, 0]: one "homework", two "lecture".
CARD = {
    "file": {"name": "week3.pdf"},
    "opening": "Lecture 3. This lecture covers kernels. Homework is due Friday.",
}


def test_group_text_has_name_and_covers_and_excludes_only_on_request():
    """A group is embedded as its name and covers; excludes only when asked; or the name alone."""
    assert group_text(SMALL_GROUPS[0]) == "Homework: homework problems"
    assert group_text(SMALL_GROUPS[2]) == "Exams: exam papers"
    assert (
        group_text(SMALL_GROUPS[2], include_excludes=True)
        == "Exams: exam papers. Excludes: homework"
    )
    assert group_text(SMALL_GROUPS[2], with_descriptions=False) == "Exams"


def test_chunks_and_mean_vector():
    """The helpers behind the whole-card mode."""
    assert chunks("abcdefg", 3) == ["abc", "def", "g"]
    assert chunks("", 3) == [""]
    assert mean_vector([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]) == pytest.approx([1.0, 1.0])


def test_embedding_whole_card_mode_embeds_every_chunk_and_averages():
    """With chunk_chars set, each piece of the card is embedded and the mean decides."""
    seen: list[list[str]] = []

    def embed(texts):
        """Groups point along one axis each; a card chunk points where its first letter says."""
        seen.append(list(texts))
        axes = {"H": [1.0, 0.0, 0.0], "L": [0.0, 1.0, 0.0], "E": [0.0, 0.0, 1.0]}
        return [axes.get(text.split(": ")[-1][:1].upper(), [0.0, 0.0, 0.0]) for text in texts]

    backend = EmbeddingBackend(embed=embed, chunk_chars=4)
    # state_text gives "note: LLLLEEEE", cut into "note", ": LL", "LLEE", "EE": two L, one E.
    result = ask(backend, "x.txt", SMALL_GROUPS, state={"note": "LLLLEEEE"})
    assert seen[1] == ["note", ": LL", "LLEE", "EE"]
    assert top(result) == "lec"
    assert sum(result.probs.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        EmbeddingBackend(embed=embed, chunk_chars=0)


def test_cosine_and_softmax():
    """The two bits of math behind the baseline."""
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([3, 4], [4, 3]) == pytest.approx(24 / 25)
    assert cosine([0, 0], [1, 1]) == 0.0  # no direction, no similarity
    assert softmax([1.0, 1.0], 0.05) == pytest.approx([0.5, 0.5])
    # A gap of 0.05 at temperature 0.05 is one unit of logit: odds of e to 1.
    assert softmax([0.55, 0.5], 0.05) == pytest.approx([math.e / (1 + math.e), 1 / (1 + math.e)])
    assert softmax([], 0.05) == []


def test_embedding_nearest_group_wins():
    """Cosines to the card are 1/sqrt(5), 2/sqrt(5) and 1/5: the lecture group is nearest."""
    backend = EmbeddingBackend(embed=FakeEmbed(), include_excludes=True)
    result = ask(backend, "week3.pdf", SMALL_GROUPS, state=CARD)
    similarities = [1 / math.sqrt(5), 2 / math.sqrt(5), 1 / 5]
    weights = [math.exp(similarity / 0.05) for similarity in similarities]
    expected = [weight / sum(weights) for weight in weights]
    assert top(result) == "lec"
    assert [result.probs[gid] for gid in ("hw", "lec", "ex")] == pytest.approx(expected)
    assert sum(result.probs.values()) == pytest.approx(1.0)


def test_embedding_leaves_excludes_out_by_default():
    """By default the Exams group embeds without "homework", so its cosine to the card is 0."""
    result = ask(EmbeddingBackend(embed=FakeEmbed()), "week3.pdf", SMALL_GROUPS, state=CARD)
    similarities = [1 / math.sqrt(5), 2 / math.sqrt(5), 0.0]
    weights = [math.exp(similarity / 0.05) for similarity in similarities]
    expected = [weight / sum(weights) for weight in weights]
    assert [result.probs[gid] for gid in ("hw", "lec", "ex")] == pytest.approx(expected)


def test_embedding_result_fields():
    """It cannot abstain, reports the local model, and has no token count."""
    backend = EmbeddingBackend(embed=FakeEmbed())
    result = ask(backend, "week3.pdf", SMALL_GROUPS, state=CARD)
    assert backend.name == "embed"
    assert result.fits_any == 1.0
    assert result.model == "embed:BAAI/bge-small-en-v1.5"
    assert result.option_order == ["hw", "lec", "ex"]
    assert result.input_tokens is None
    assert result.latency_ms >= 0


def test_embedding_reads_the_card_through_state_text():
    """The text that gets embedded for the file is the flattened card."""
    fake = FakeEmbed()
    ask(EmbeddingBackend(embed=fake), "week3.pdf", SMALL_GROUPS, state=CARD)
    assert fake.calls[-1] == [state_text(CARD)]


def test_embedding_names_only_mode_embeds_just_the_names():
    """With descriptions off, the group texts are the bare names."""
    fake = FakeEmbed()
    backend = EmbeddingBackend(embed=fake)
    result = ask(backend, "week3.pdf", SMALL_GROUPS, state=CARD, with_descriptions=False)
    assert fake.calls[0] == ["Homework", "Lecture notes", "Exams"]
    # Names embed as [1, 0, 0], [0, 1, 0] and [0, 0, 1]: lecture is still nearest to [1, 2, 0].
    assert top(result) == "lec"
    assert sum(result.probs.values()) == pytest.approx(1.0)


def test_embedding_caches_group_vectors_per_group_set():
    """Groups are embedded once per set of texts. Only the card is embedded on later calls."""
    fake = FakeEmbed()
    backend = EmbeddingBackend(embed=fake)
    ask(backend, "one.pdf", SMALL_GROUPS, state=CARD)
    assert len(fake.calls) == 2  # the groups, then the card
    ask(backend, "two.pdf", SMALL_GROUPS, state=CARD)
    assert len(fake.calls) == 3  # the card only
    ask(backend, "three.pdf", SMALL_GROUPS, state=CARD, with_descriptions=False)
    assert len(fake.calls) == 5  # other texts for the same groups: embedded again
    ask(backend, "four.pdf", SMALL_GROUPS[:2], state=CARD)
    assert len(fake.calls) == 7  # another group set
    ask(backend, "five.pdf", SMALL_GROUPS, state=CARD)
    assert len(fake.calls) == 8  # the first set is still cached


def test_embedding_latency_leaves_out_the_one_time_group_embedding(monkeypatch):
    """With a clock that ticks one second per embed call, a sort takes one second: the card.

    The first sort also embeds the groups, but that happens before the timer starts.
    """
    clock = {"now": 0.0}

    def ticking(texts):
        """Embed with count_words, and move the clock one second forward."""
        clock["now"] += 1.0
        return [count_words(text) for text in texts]

    monkeypatch.setattr(embed_module, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))
    backend = EmbeddingBackend(embed=ticking)
    first = ask(backend, "one.pdf", SMALL_GROUPS, state=CARD)
    second = ask(backend, "two.pdf", SMALL_GROUPS, state=CARD)
    assert first.latency_ms == pytest.approx(1000)
    assert second.latency_ms == pytest.approx(1000)


def test_embedding_temperature_flattens_or_sharpens():
    """A high temperature pulls the probabilities together. It must be above 0."""
    flat = ask(EmbeddingBackend(embed=FakeEmbed(), temperature=100.0), "f", SMALL_GROUPS, CARD)
    sharp = ask(EmbeddingBackend(embed=FakeEmbed()), "f", SMALL_GROUPS, CARD)
    assert max(flat.probs.values()) < 0.34
    assert max(sharp.probs.values()) > 0.99
    with pytest.raises(ValueError):
        EmbeddingBackend(embed=FakeEmbed(), temperature=0)


def test_embedding_builds_the_fastembed_model_lazily(monkeypatch, tmp_path):
    """No embed function given: the model is built on the first sort, once, with our settings."""
    built = []

    class FakeTextEmbedding:
        """Stands in for fastembed.TextEmbedding, so no model is downloaded."""

        def __init__(self, model_name, cache_dir=None):
            """Record how the model was built."""
            built.append((model_name, cache_dir))

        def embed(self, documents):
            """Yield numpy vectors one by one, as fastembed does."""
            for text in documents:
                yield np.array(count_words(text), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    backend = EmbeddingBackend(cache_dir=tmp_path / "models")
    assert built == []
    first = ask(backend, "week3.pdf", SMALL_GROUPS, state=CARD)
    ask(backend, "again.pdf", SMALL_GROUPS, state=CARD)
    assert built == [(DEFAULT_MODEL_NAME, str(tmp_path / "models"))]
    assert top(first) == "lec"
    assert all(type(prob) is float for prob in first.probs.values())


def test_embedding_failure_is_a_backend_error():
    """A model that fails, or returns the wrong number of vectors, is never an answer."""

    def broken(texts):
        """An embed function that always fails."""
        raise RuntimeError("model file is damaged")

    with pytest.raises(BackendError, match="embedding failed"):
        ask(EmbeddingBackend(embed=broken), "week3.pdf", SMALL_GROUPS, state=CARD)
    with pytest.raises(BackendError, match="vectors"):
        ask(EmbeddingBackend(embed=lambda texts: [[1.0]]), "week3.pdf", SMALL_GROUPS, state=CARD)


# LLM baseline


def reply(text='{"group": "Homework"}', stop_reason="end_turn", **extra):
    """A response shaped like the Anthropic SDK's: an empty thinking block, then the text."""
    fields = {
        "content": [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        "stop_reason": stop_reason,
        "usage": SimpleNamespace(input_tokens=3210, output_tokens=18),
        "model": "claude-fable-5-2",
        "_request_id": "req_test_1",
    }
    fields.update(extra)
    return SimpleNamespace(**fields)


class FakeClient:
    """Stands in for anthropic.Anthropic: returns one canned response and records each call."""

    def __init__(self, response=None, error=None):
        """Answer every call with this response, or raise this error."""
        self.calls = []
        self._response = response
        self._error = error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        """Record the request, then answer or fail as set up."""
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


HW_CARD = {
    "file": {"name": "hw3_solution.py", "path": "hw3/hw3_solution.py", "type": "Python code"},
    "title": "Homework 3: Kernel Methods",
    "outline": ["rbf_kernel()", "train_svm()"],
    "opening": "Implement the RBF kernel and train an SVM.",
}


def test_llm_happy_path_puts_everything_on_the_chosen_group():
    """The model names one group. That group gets probability 1.0."""
    client = FakeClient(reply('{"group": "Homework"}'))
    backend = llm.LLMBackend(client=client)
    result = ask(backend, "hw3/hw3_solution.py", state=HW_CARD)
    assert backend.name == "llm"
    assert result.probs == {gid: 1.0 if gid == "homework" else 0.0 for gid in EVAL_IDS}
    assert result.fits_any == 1.0
    assert result.input_tokens == 3210
    assert result.model == "claude-fable-5-2"
    assert result.request_id == "req_test_1"
    assert result.option_order == EVAL_IDS
    assert result.latency_ms >= 0


def test_llm_request_carries_the_card_the_groups_and_a_forced_choice():
    """One request: the pinned model, the card text, every description, and a schema that
    allows only the group names."""
    client = FakeClient(reply())
    ask(llm.LLMBackend(client=client), "hw3/hw3_solution.py", state=HW_CARD)
    (call,) = client.calls
    assert call["model"] == llm.DEFAULT_MODEL == "claude-fable-5-2"
    assert call["max_tokens"] == 4096
    assert call["system"] == llm.SYSTEM_PROMPT
    # Thinking and sampling settings are rejected by current models, so none are sent.
    assert set(call) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert call["output_config"]["effort"] == llm.EFFORT
    answer = call["output_config"]["format"]
    assert answer["type"] == "json_schema"
    assert answer["schema"]["properties"]["group"]["enum"] == [g.name for g in EVAL_GROUPS]
    assert answer["schema"]["required"] == ["group"]
    assert answer["schema"]["additionalProperties"] is False
    (message,) = call["messages"]
    assert message["role"] == "user"
    assert state_text(HW_CARD) in message["content"]
    for group in EVAL_GROUPS:
        assert f"- {group.name}\n  covers: {group.covers}" in message["content"]
    assert "  excludes: homework and exams" in message["content"]
    # The question is the one Jev is asked, word for word, plus how to answer.
    assert message["content"].endswith(f"{GROUP_INSTRUCTIONS} Answer with the group's name.")


def test_llm_names_only_mode_leaves_the_descriptions_out():
    """With descriptions off, the prompt lists the names and nothing else about the groups."""
    client = FakeClient(reply())
    ask(llm.LLMBackend(client=client), "hw3.py", state=HW_CARD, with_descriptions=False)
    content = client.calls[0]["messages"][0]["content"]
    assert "- Homework\n- Exams" in content
    assert "covers:" not in content
    assert "excludes:" not in content


def test_llm_model_and_max_tokens_can_be_set():
    """The constructor's model and max_tokens are what the request uses."""
    client = FakeClient(reply(model="claude-haiku-5-5"))
    backend = llm.LLMBackend(model="claude-haiku-5-5", client=client, max_tokens=512)
    ask(backend, "hw3.py", state=HW_CARD)
    assert client.calls[0]["model"] == "claude-haiku-5-5"
    assert client.calls[0]["max_tokens"] == 512


def test_llm_result_names_the_model_that_answered():
    """The result records the model the API reports, not the one that was asked for."""
    client = FakeClient(reply(model="model-as-reported"))
    result = ask(llm.LLMBackend(model="model-as-asked", client=client), "hw3.py", state=HW_CARD)
    assert result.model == "model-as-reported"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (reply('{"group": "Recipes"}'), "not one of the groups"),
        (reply('{"group": "homework"}'), "not one of the groups"),  # names are matched exactly
        (reply("Homework"), "could not read a group"),  # not JSON
        (reply('{"answer": "Homework"}'), "could not read a group"),
        (reply('{"group": ["Homework"]}'), "could not read a group"),
        (reply('["Homework"]'), "could not read a group"),
        (reply(stop_reason="max_tokens"), "cut off"),
        (reply(content=[SimpleNamespace(type="thinking", thinking="")]), "no text"),
        (
            reply(content=[], stop_reason="refusal", stop_details=SimpleNamespace(category="bio")),
            "declined to answer",
        ),
    ],
)
def test_llm_bad_answer_is_a_backend_error(response, message):
    """Anything but exactly one group name is a failure, never an answer."""
    backend = llm.LLMBackend(client=FakeClient(response))
    with pytest.raises(BackendError, match=message):
        ask(backend, "hw3.py", state=HW_CARD)


def test_llm_api_error_is_a_backend_error():
    """An error from the Anthropic SDK comes out as a BackendError."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(error=anthropic.APIConnectionError(request=request))
    with pytest.raises(BackendError, match="Anthropic API error"):
        ask(llm.LLMBackend(client=client), "hw3.py", state=HW_CARD)


def sdk_client(handler):
    """A real Anthropic client whose requests go to `handler` instead of the network."""
    return anthropic.Anthropic(
        api_key="test-key-not-a-real-one",
        base_url="https://api.anthropic.com",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def api_message(**fields):
    """A Messages API response body, as JSON: an empty thinking block, then the answer."""
    body = {
        "id": "msg_test_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-fable-5-2",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": '{"group": "Homework"}'},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3210, "output_tokens": 44},
    }
    body.update(fields)
    return body


def test_llm_works_through_the_real_sdk_with_a_fake_transport():
    """The request the Anthropic SDK really sends, and the response it really parses.

    The other tests use a stand-in client that would take any argument. Here the real client
    runs over a fake transport, so a wrong argument name or a wrong response field would show.
    """
    sent = []

    def answer(request):
        """Record the HTTP request and answer like the Messages API."""
        sent.append(request)
        return httpx2.Response(200, headers={"request-id": "req_test_2"}, json=api_message())

    with sdk_client(answer) as client:
        result = ask(llm.LLMBackend(client=client), "hw3/hw3_solution.py", state=HW_CARD)
    assert result.probs == {gid: 1.0 if gid == "homework" else 0.0 for gid in EVAL_IDS}
    assert result.input_tokens == 3210
    assert result.model == "claude-fable-5-2"
    assert result.request_id == "req_test_2"
    (request,) = sent
    assert request.url.path == "/v1/messages"
    body = json.loads(request.content)
    assert set(body) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert body["model"] == llm.DEFAULT_MODEL
    assert body["output_config"] == {"effort": llm.EFFORT, "format": llm.answer_format(EVAL_GROUPS)}
    (message,) = body["messages"]
    assert message["role"] == "user"
    assert state_text(HW_CARD) in message["content"]


def test_llm_refusal_and_http_error_through_the_real_sdk():
    """A refusal (HTTP 200) and a rejected request (HTTP 400), as the real SDK reports them."""

    def refuse(request):
        """Answer like a request the safety classifiers declined: no content at all."""
        details = {"type": "refusal", "category": "frontier_llm", "explanation": None}
        body = api_message(content=[], stop_reason="refusal", stop_details=details)
        return httpx2.Response(200, json=body)

    def reject(request):
        """Answer like a request the API turns down."""
        error = {"type": "invalid_request_error", "message": "retention is not enabled"}
        return httpx2.Response(400, json={"type": "error", "error": error})

    refused = pytest.raises(BackendError, match="refusal, category: frontier_llm")
    with sdk_client(refuse) as client, refused:
        ask(llm.LLMBackend(client=client), "hw3.py", state=HW_CARD)
    rejected = pytest.raises(BackendError, match="retention is not enabled")
    with sdk_client(reject) as client, rejected as failure:
        ask(llm.LLMBackend(client=client), "hw3.py", state=HW_CARD)
    assert "test-key" not in str(failure.value)


def test_llm_without_a_key_or_a_client_is_unavailable(monkeypatch):
    """No key and no client: the constructor says which key is missing."""
    asked = []

    def no_key(name, start=None):
        """A key lookup that records what was asked for and finds nothing (returns None)."""
        asked.append(name)

    monkeypatch.setattr(llm, "env_value", no_key)
    with pytest.raises(BackendError, match="ANTHROPIC_API_KEY"):
        llm.LLMBackend()
    assert asked == ["ANTHROPIC_API_KEY"]


def test_llm_builds_its_client_from_the_key(monkeypatch):
    """With a key and no client, the Anthropic client is built with that key."""
    built = []

    def fake_anthropic(**kwargs):
        """Record how the client was built and hand back a fake."""
        built.append(kwargs)
        return FakeClient(reply())

    monkeypatch.setattr(llm, "env_value", lambda name, start=None: "not-a-real-key")
    monkeypatch.setattr(llm.anthropic, "Anthropic", fake_anthropic)
    backend = llm.LLMBackend()
    assert built == [{"api_key": "not-a-real-key"}]
    assert top(ask(backend, "hw3.py", state=HW_CARD)) == "homework"


def test_llm_with_a_client_never_looks_for_a_key(monkeypatch):
    """An injected client is enough: no key lookup happens."""

    def fail(name, start=None):
        """A key lookup that must not be reached."""
        raise AssertionError("looked for a key")

    monkeypatch.setattr(llm, "env_value", fail)
    llm.LLMBackend(client=FakeClient(reply()))


# Baseline answers through the placement rules


def placed(result):
    """Where the default cutoffs put a result: (group id, or None for Unsorted; close call)."""
    prediction = Prediction(
        file_id=result.file_id, gold=None, probs=result.probs, fits_any=result.fits_any
    )
    return outcome(prediction, Cutoffs.from_config(Config()))


def test_baseline_answers_under_the_default_cutoffs():
    """A one-hot answer is placed with no close call flag. No rules match ends up Unsorted."""
    assert placed(ask(RulesBackend(), "syllabus.pdf")) == ("logistics", False)
    assert placed(ask(RulesBackend(), "kernel_methods_2006.pdf")) == (None, False)
    picked = ask(llm.LLMBackend(client=FakeClient(reply())), "hw3.py", state=HW_CARD)
    assert placed(picked) == ("homework", False)
    # The embedding baseline is only placed when one group is far enough ahead of the rest.
    sharp = ask(EmbeddingBackend(embed=FakeEmbed()), "week3.pdf", SMALL_GROUPS, state=CARD)
    flat = EmbeddingBackend(embed=FakeEmbed(), temperature=100.0)
    assert placed(sharp) == ("lec", False)
    assert placed(ask(flat, "week3.pdf", SMALL_GROUPS, state=CARD)) == (None, False)
