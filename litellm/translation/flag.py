"""The single v2 opt-in.

One env var, ``LLM_TRANSLATION_V2``, turns v2 translation on for every ported
provider at once. Which providers are ported is a fact of the code (the
serializer registry in ``engine.pipeline``), not configuration, so there is
nothing else to flip. This is the only place the environment is read; everything
downstream takes the resulting bool as a parameter.
"""

from __future__ import annotations

import os

TRANSLATION_V2_ENV = "LLM_TRANSLATION_V2"

_TRUTHY = frozenset({"true", "1"})


def is_translation_v2_enabled() -> bool:
    return os.getenv(TRANSLATION_V2_ENV, "").strip().lower() in _TRUTHY
