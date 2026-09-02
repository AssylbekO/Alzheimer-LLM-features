"""OpenAI-compatible provider for litellm.vse.cz, hardened for a large model
behind a reverse proxy.

Three things break with the plain LocalProvider on a 122B model:

1. 504 Bad Gateway — a non-streamed multi-second generation looks idle to the
   proxy. Streaming keeps bytes flowing.
2. Empty response / "Invalid JSON response:" — Qwen3.x reasons internally
   before answering and can spend the whole token budget doing it.
   extra_body={"think": False} turns that off; stripping <think>...</think>
   is the backup for endpoints that ignore the flag.
3. The base library retries rate limits only — not timeouts, 5xx, or empty
   bodies.

This subclass overrides the single method every call funnels through
(_chat_json) and the raw HTTP call beneath it (_raw_call). Everything else
about the library is unchanged.
"""
from __future__ import annotations

import json
import re
import time

import openai
from openai import BadRequestError
from llm_feature_gen.providers.local_provider import LocalProvider

from . import config

THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class StreamingLocalProvider(LocalProvider):
    """LocalProvider + streaming + think-off + retries on transient failures."""

    def __init__(self, *a, stream: bool = True, **kw):
        super().__init__(*a, **kw)
        self.stream = stream
        self.client = openai.OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=config.TIMEOUT_S
        )
        self._think_supported = True   # flipped off if the endpoint rejects the field

    def _raw_call(self, model, messages, json_mode):
        kw = dict(model=model, messages=messages,
                  temperature=self.temperature, max_tokens=self.max_tokens)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if self._think_supported:
            kw["extra_body"] = {"think": False}

        if not self.stream:
            return self.client.chat.completions.create(**kw).choices[0].message.content or ""

        parts = []
        for ev in self.client.chat.completions.create(stream=True, **kw):
            if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                parts.append(ev.choices[0].delta.content)
        return "".join(parts)

    def _chat_json(self, deployment_name, system_prompt, user_content, json_mode=False):
        if json_mode and "JSON" not in system_prompt:
            system_prompt += " Respond in strict JSON format."
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}]

        last, backoff = None, 3
        for attempt in range(self.max_retries):
            try:
                text = self._raw_call(deployment_name, messages, json_mode)
                text = THINK_TAG.sub("", text).strip()
                if not text:
                    raise ValueError("empty completion")
                try:
                    return json.loads(text)
                except Exception:
                    got = self._extract_json(text)      # handles ```json fences
                    if got:
                        return {"features": got} if isinstance(got, list) else got
                    raise ValueError(f"unparseable: {text[:200]}")

            except BadRequestError as e:
                msg = str(e)
                if self._think_supported and "think" in msg:
                    self._think_supported = False       # endpoint rejects the flag
                    continue
                if json_mode and "json_object" in msg:
                    json_mode = False
                    continue
                raise

            except Exception as e:                      # timeout, 5xx, empty, unparseable
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
        raise RuntimeError(f"failed after {self.max_retries} attempts: {last}")


def make_provider(model: str, max_retries: int = 4, stream: bool = True) -> StreamingLocalProvider:
    """Build a ready-to-use provider for `model` (config.MODEL_A / MODEL_B, or
    any other model name available on the endpoint)."""
    return StreamingLocalProvider(
        base_url=config.BASE_URL, api_key=config.API_KEY, default_text_model=model,
        temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
        max_retries=max_retries, stream=stream,
    )
