"""The V2 extraction prompt — single source of truth.

Label-blind and domain-blind: the LLM counts observable linguistic events
and quotes evidence, and is never told (even implicitly, by naming what NOT
to infer) that this is a cognitive-screening study.

Every notebook that extracts or compares runs must import V2_PROMPT and
PROMPT_SHA from here, and must not paste the text in. If the prompt ever
needs to change, it changes in exactly one place, and every notebook that
asserts the old hash will fail loudly instead of silently comparing runs
made under two different prompts.
"""
from __future__ import annotations

import hashlib

V2_PROMPT = r"""You are annotating a transcript of spontaneous Czech speech. A person was shown a drawing of a
lakeshore scene and asked to describe it aloud. The text is an automatic transcription.

Your job is to COUNT observable linguistic events and QUOTE the evidence for each count.
You are an annotator, not an evaluator.

RULES
- Quote evidence verbatim from the transcript, in Czech. Never translate or paraphrase.
- If a category has no instances, return an empty list. Empty is a valid answer.
- Never infer anything about the speaker: not their health, ability, intelligence, age,
  education or state of mind. Describe the language, never the person.
- Do not compare this speaker to anyone else or to any norm.
- Count occurrences, not impressions. Two hedges in one sentence are two entries.
- Return only JSON.

CATEGORIES
1. named_entities - distinct objects, creatures or people explicitly named. Lemmatise and list
   each distinct entity exactly once (a set, not a list of mentions).
2. specific_action_verbs - verbs naming a particular manner of action. List every occurrence.
3. generic_verbs - verbs of bare existence, possession, location or unspecified movement.
4. complete_propositions - integer count of clauses with an explicit subject and predicate
   plus at least one further argument or adjunct.
5. locative_expressions - phrases placing something somewhere. List every occurrence.
6. regions_referenced - distinct regions among "water", "land", "sky".
7. hedge_spans - expressions of uncertainty. List every occurrence.
8. deictic_spans - references such as "there", "that thing" that substitute for naming.
9. metacomment_spans - remarks about the speaker's own describing, remembering, or the task.
10. repeated_content_lemmas - content words used more than once, with counts.
11. self_corrections - integer count.
12. diminutive_or_affective_forms - diminutive or affectionate noun forms.
13. quantity_expressions - numerals or quantifiers applied to things in the scene.

Return exactly this JSON and nothing else:
{
  "named_entities": [],
  "specific_action_verbs": [],
  "generic_verbs": [],
  "complete_propositions": 0,
  "locative_expressions": [],
  "regions_referenced": [],
  "hedge_spans": [],
  "deictic_spans": [],
  "metacomment_spans": [],
  "repeated_content_lemmas": [{"lemma": "", "count": 0}],
  "self_corrections": 0,
  "diminutive_or_affective_forms": [],
  "quantity_expressions": []
}"""

PROMPT_SHA = hashlib.sha256(V2_PROMPT.encode()).hexdigest()[:16]

_BANNED_TERMS = [
    "dementia", "alzheimer", "cognitive", "impair", "patient", "diagnos",
    "control group", "demence", "pacient",
]


def assert_domain_blind() -> None:
    """The prompt must never leak that this is a cognitive-screening study —
    not even by naming what NOT to infer, since that still tells the model
    what the study is about."""
    low = V2_PROMPT.lower()
    for term in _BANNED_TERMS:
        assert term not in low, f"prompt leaks domain term: {term!r}"


def assert_prompt_hash(expected: str) -> None:
    """Call this at the top of every notebook that compares two extraction
    runs, or that loads a frozen pipeline. A mismatch means the prompt has
    changed since the runs / artifact being used were produced, and the
    comparison (or the frozen features) is no longer valid."""
    assert PROMPT_SHA == expected, (
        f"prompt hash mismatch: got {PROMPT_SHA}, expected {expected}. "
        "The prompt text has changed since the runs being compared were "
        "produced — do not proceed until this is resolved."
    )
