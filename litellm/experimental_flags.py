"""Internal experimental flags (import-safe)."""
import os
LITELLM_ENABLE_PARALLEL_ACOMPLETIONS = os.getenv('LITELLM_ENABLE_PARALLEL_ACOMPLETIONS', '0')
