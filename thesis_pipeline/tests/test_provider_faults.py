"""Fault-injection tests for StreamingLocalProvider. Never touches the
network — openai.OpenAI is monkeypatched with a scripted fake client that
can simulate 504s, timeouts, empty bodies, <think> tags, ```json fences,
and an endpoint that rejects the extra_body={"think": False} field.
"""
from __future__ import annotations

import json
import types

import httpx
import openai
import pytest
from openai import BadRequestError

import thesis_pipeline.provider as provider_mod

try:
    import llm_feature_gen.providers.local_provider as _lp
except ImportError:  # pragma: no cover - only relevant if the import shape changes
    _lp = None


GOOD = json.dumps({
    "named_entities": ["pes"], "specific_action_verbs": ["běží"],
    "generic_verbs": ["je"], "complete_propositions": 2,
    "locative_expressions": ["na břehu"], "regions_referenced": ["land"],
    "hedge_spans": [], "deictic_spans": ["tam"], "metacomment_spans": [],
    "repeated_content_lemmas": [{"lemma": "pes", "count": 2}],
    "self_corrections": 0, "diminutive_or_affective_forms": [],
    "quantity_expressions": [],
}, ensure_ascii=False)


class _Delta:
    def __init__(self, c):
        self.content = c


class _Choice:
    def __init__(self, c):
        self.delta = _Delta(c)
        self.message = types.SimpleNamespace(content=c)


class _Ev:
    def __init__(self, c):
        self.choices = [_Choice(c)]


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "http://x/v1/chat/completions"))


@pytest.fixture
def fake_client(monkeypatch):
    calls = {"n": 0, "think_seen": 0, "stream_seen": 0}
    script: dict = {}

    class FakeCompletions:
        def create(self, **kw):
            calls["n"] += 1
            n = calls["n"]
            if kw.get("extra_body", {}).get("think") is False:
                calls["think_seen"] += 1
            streaming = kw.get("stream", False)
            if streaming:
                calls["stream_seen"] += 1
            behaviour = script.get(n, "good")
            if behaviour == "504":
                raise openai.APIStatusError("Bad Gateway", response=_resp(504), body=None)
            if behaviour == "timeout":
                raise openai.APITimeoutError(request=None)
            if behaviour == "empty":
                body = ""
            elif behaviour == "think":
                body = "<think>hmm let me count</think>" + GOOD
            elif behaviour == "fence":
                body = "```json\n" + GOOD + "\n```"
            elif behaviour == "nothink_reject":
                raise BadRequestError("unknown field: think", response=_resp(400), body=None)
            elif behaviour == "dead":
                raise openai.APITimeoutError(request=None)
            else:
                body = GOOD
            if streaming:
                return iter([_Ev(body[i:i + 7]) for i in range(0, len(body), 7)] or [_Ev("")])
            return types.SimpleNamespace(choices=[_Choice(body)])

    class FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(provider_mod.openai, "OpenAI", FakeClient)
    if _lp is not None and hasattr(_lp, "OpenAI"):
        monkeypatch.setattr(_lp, "OpenAI", FakeClient)

    # the retry path sleeps 3s, 6s, 12s between attempts — skip the wall-clock
    # wait, the scenarios under test are about control flow, not timing.
    monkeypatch.setattr(provider_mod.time, "sleep", lambda *_a, **_k: None)

    return calls, script


def _fresh():
    return provider_mod.StreamingLocalProvider(
        base_url="http://x/v1", api_key="k", default_text_model="m",
        temperature=0.0, max_tokens=4096, max_retries=4, stream=True,
    )


def test_happy_path_streams_and_disables_think(fake_client):
    calls, _script = fake_client
    p = _fresh()
    r = p.text_features(["text"], prompt="P")[0]
    assert r["complete_propositions"] == 2
    assert calls["stream_seen"] == 1
    assert calls["think_seen"] == 1


def test_recovers_from_two_consecutive_504s(fake_client):
    _calls, script = fake_client
    script.update({1: "504", 2: "504"})
    p = _fresh()
    r = p.text_features(["text"], prompt="P")[0]
    assert r["complete_propositions"] == 2


def test_retries_past_empty_body(fake_client):
    _calls, script = fake_client
    script.update({1: "empty"})
    p = _fresh()
    r = p.text_features(["text"], prompt="P")[0]
    assert r["complete_propositions"] == 2


def test_strips_think_tags(fake_client):
    _calls, script = fake_client
    script.update({1: "think"})
    p = _fresh()
    r = p.text_features(["text"], prompt="P")[0]
    assert r["complete_propositions"] == 2


def test_parses_json_code_fence(fake_client):
    _calls, script = fake_client
    script.update({1: "fence"})
    p = _fresh()
    r = p.text_features(["text"], prompt="P")[0]
    assert r["complete_propositions"] == 2


def test_falls_back_when_endpoint_rejects_think_field(fake_client):
    _calls, script = fake_client
    script.update({1: "nothink_reject"})
    p = _fresh()
    r = p.text_features(["text"], prompt="P")[0]
    assert r["complete_propositions"] == 2
    assert p._think_supported is False


def test_permanent_failure_raises_after_max_retries(fake_client):
    calls, script = fake_client
    script.update({i: "dead" for i in range(1, 20)})
    p = _fresh()
    with pytest.raises(RuntimeError):
        p.text_features(["text"], prompt="P")
    assert calls["n"] == 4
