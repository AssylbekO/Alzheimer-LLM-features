"""Provider-logic and resume tests exercise pure code paths and never call a
real endpoint, but `thesis_pipeline.config` raises at import if
LOCAL_OPENAI_API_KEY is unset. Provide a dummy so the suite runs without a
real .env; a test that actually needed a live key would fail on the network
call, not here.
"""
import os

os.environ.setdefault("LOCAL_OPENAI_API_KEY", "test-dummy-key")
