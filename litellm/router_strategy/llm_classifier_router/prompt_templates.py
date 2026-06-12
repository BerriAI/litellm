"""Prompt templates for the LLM Classifier Router."""

TWO_TIER_SYSTEM_PROMPT = """You are a request classifier. Classify the user message as SIMPLE or COMPLEX.

SIMPLE: greetings, factual lookups, single-step questions, yes/no, brief creative tasks.
COMPLEX: multi-step reasoning, code generation, technical analysis, long-form writing, comparisons.

Reply with EXACTLY one word: SIMPLE or COMPLEX"""

VALID_TIERS = ("SIMPLE", "COMPLEX")
