"""Shared code for the V2 LLM-feature extraction thesis pipeline.

Import from submodules directly, e.g.:

    from thesis_pipeline import config, prompt, provider, data
    from thesis_pipeline import extraction, features, stability, evaluation, freeze, plotting

This package exists to hold code that is otherwise pasted, byte-for-byte,
into more than one notebook (08, 09, and every notebook after them). If you
find yourself copying a function out of a notebook into another notebook,
it belongs in here instead.
"""
