"""Regex-based task classification engine for auto-routing.

Classifies user messages into task categories (heartbeat, coding, reasoning, etc.)
to determine which model tier should handle the request.

Ported from ClawRouter's classifier with Python 3.9 compatibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class TaskCategory(str, Enum):
    """Task categories for routing decisions."""

    HEARTBEAT = "heartbeat"
    SIMPLE_CHAT = "simple-chat"
    LOOKUP = "lookup"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CODING = "coding"
    CREATIVE = "creative"
    REASONING = "reasoning"
    ANALYSIS = "analysis"


@dataclass(frozen=True)
class ClassificationResult:
    """Result of classifying a user message."""

    category: TaskCategory
    confidence: float  # 0.0 to 1.0
    matched_pattern: str  # The pattern that matched (for debugging)


@dataclass(frozen=True)
class ClassificationRule:
    """A single classification rule: pattern -> category with confidence."""

    pattern: "re.Pattern[str]"
    category: TaskCategory
    confidence: float
    description: str


# Ordered rules: first match wins. More specific patterns come first.
DEFAULT_PATTERNS: Sequence[ClassificationRule] = [
    # --- Heartbeat: OpenClaw system heartbeats ---
    ClassificationRule(
        pattern=re.compile(
            r"(read\s+heartbeat\.md\b|"
            r"\bheartbeat_ok\b|"
            r"\breply\s+heartbeat_ok\b)",
            re.IGNORECASE,
        ),
        category=TaskCategory.HEARTBEAT,
        confidence=0.98,
        description="OpenClaw heartbeat prompt or token",
    ),
    # --- Heartbeat: very short pings ---
    ClassificationRule(
        pattern=re.compile(
            r"^(hi|hey|hello|ping|test|yo|sup|hola|ok|okay|k|yes|no|thanks|thx|ty|"
            r"are you there\??|you there\??|alive\??|awake\??)\.?!?\s*$",
            re.IGNORECASE,
        ),
        category=TaskCategory.HEARTBEAT,
        confidence=0.95,
        description="Short pings and greetings",
    ),
    # --- Reasoning: math, proofs, logic ---
    ClassificationRule(
        pattern=re.compile(
            r"(prove\s+that|proof\s+of|explain\s+why|what\s+is\s+the\s+relationship\s+between|"
            r"derive\s+|theorem|lemma|corollary|axiom|mathematical(ly)?|"
            r"integral\s+of|derivative\s+of|solve\s+(for|the\s+equation)|"
            r"calculate\s+the\s+(probability|expected|variance)|"
            r"if\s+and\s+only\s+if|necessary\s+and\s+sufficient|"
            r"by\s+contradiction|by\s+induction|"
            r"what\s+would\s+happen\s+if|consider\s+the\s+(case|scenario)|"
            r"step[\s-]by[\s-]step\s+(reason|think|logic|analys)|"
            r"multi[\s-]?step|chain[\s-]of[\s-]thought)",
            re.IGNORECASE,
        ),
        category=TaskCategory.REASONING,
        confidence=0.90,
        description="Math, proofs, and multi-step logic",
    ),
    # --- Analysis: compare, evaluate, research ---
    ClassificationRule(
        pattern=re.compile(
            r"(^analyze\b|^compare\b|^evaluate\b|^assess\b|^review\b|^critique\b|"
            r"^research\b|^investigate\b|^examine\b|"
            r"pros?\s+and\s+cons?|trade[\s-]?offs?|advantages?\s+and\s+disadvantages?|"
            r"in[\s-]?depth\s+(analysis|review|look)|"
            r"strengths?\s+and\s+weaknesses?|"
            r"comprehensive(ly)?\s+(review|analysis|overview|look))",
            re.IGNORECASE,
        ),
        category=TaskCategory.ANALYSIS,
        confidence=0.85,
        description="Analysis, comparison, and evaluation",
    ),
    # --- Coding: code blocks, programming keywords, file paths ---
    ClassificationRule(
        pattern=re.compile(
            r"(```|`[^`]+`|"
            r"def\s+\w+|class\s+\w+|function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|"
            r"import\s+\w+|from\s+\w+\s+import|require\s*\(|"
            r"\w+\.\w+\.(py|js|ts|go|rs|java|cpp|c|rb|sh|yaml|json|toml|md)|"
            r"(write|create|build|implement|fix|debug|refactor)\s+.{0,30}(function|class|method|api|endpoint|script|program|module|component|test|database|service|handler|middleware|route|model|schema|interface|library|package|binary|tree)|"
            r"(bug|error|exception|traceback|stack\s*trace|segfault|core\s*dump)|"
            r"(npm|pip|cargo|go\s+get|apt|brew|yarn|pnpm)\s+install|"
            r"git\s+(commit|push|pull|merge|rebase|checkout|branch|clone|diff|log|stash)|"
            r"(docker|kubectl|terraform|ansible)\s+|"
            r"(SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE)\s+|"
            r"(regex|regexp|pattern\s+match)|"
            r"(localhost|127\.0\.0\.1|0\.0\.0\.0):\d+|"
            r"https?://\S+/api/)",
            re.IGNORECASE,
        ),
        category=TaskCategory.CODING,
        confidence=0.85,
        description="Code, programming, and technical tasks",
    ),
    # --- Translation ---
    ClassificationRule(
        pattern=re.compile(
            r"(translate\s+(this|the|following|it|from|to|into)|"
            r"(in|to|into)\s+(spanish|french|german|chinese|japanese|korean|"
            r"portuguese|italian|russian|arabic|hindi|turkish|dutch|swedish|"
            r"polish|czech|thai|vietnamese|indonesian|malay|tagalog|"
            r"mandarin|cantonese|english)\b|"
            r"how\s+do\s+you\s+say\s+.+\s+in\s+\w+)",
            re.IGNORECASE,
        ),
        category=TaskCategory.TRANSLATION,
        confidence=0.90,
        description="Translation between languages",
    ),
    # --- Summarization ---
    ClassificationRule(
        pattern=re.compile(
            r"(^summarize\b|^summary\s+of\b|^tldr\b|^tl;?dr\b|"
            r"give\s+me\s+(a\s+)?summary|"
            r"summarize\s+(this|the|following|it)|"
            r"can\s+you\s+summarize|"
            r"brief(ly)?\s+(summarize|overview|recap)|"
            r"key\s+(points?|takeaways?)\s+(from|of))",
            re.IGNORECASE,
        ),
        category=TaskCategory.SUMMARIZATION,
        confidence=0.90,
        description="Summarization and condensation",
    ),
    # --- Creative: writing, composing, drafting ---
    ClassificationRule(
        pattern=re.compile(
            r"(^write\s+(me\s+)?(a|an|the|some)\b|"
            r"^compose\b|^draft\b|^create\s+(a|an)\s+(story|poem|essay|article|blog|email|letter|speech|song)|"
            r"(write|tell)\s+(me\s+)?(a\s+)?(story|poem|joke|haiku|limerick|sonnet)|"
            r"(creative|fiction|narrative|prose)\s+(writing|piece|work)|"
            r"brainstorm\s+(ideas?|names?|titles?|concepts?)|"
            r"(rewrite|rephrase|paraphrase)\s+(this|the|following|it))",
            re.IGNORECASE,
        ),
        category=TaskCategory.CREATIVE,
        confidence=0.85,
        description="Creative writing and composition",
    ),
    # --- Lookup: factual questions ---
    ClassificationRule(
        pattern=re.compile(
            r"(^what\s+is\b|^what\s+are\b|^who\s+(is|was|are|were)\b|"
            r"^when\s+(did|was|is|will)\b|^where\s+(is|was|are|were)\b|"
            r"^how\s+(many|much|old|long|far|tall|big)\b|"
            r"^define\b|^definition\s+of\b|"
            r"^what\s+does\s+\w+\s+mean|"
            r"^is\s+it\s+true\s+that\b|"
            r"^(capital|population|currency)\s+of\b)",
            re.IGNORECASE,
        ),
        category=TaskCategory.LOOKUP,
        confidence=0.80,
        description="Factual questions and lookups",
    ),
    # --- Simple chat: short casual messages (fallback before default) ---
    ClassificationRule(
        pattern=re.compile(
            r"^.{1,80}$",  # Short messages under 80 chars that didn't match above
            re.IGNORECASE | re.DOTALL,
        ),
        category=TaskCategory.SIMPLE_CHAT,
        confidence=0.60,
        description="Short casual messages (length-based fallback)",
    ),
]


def classify_task(
    text: str,
    custom_rules: Optional[Sequence[ClassificationRule]] = None,
) -> ClassificationResult:
    """Classify a user message into a task category.

    Args:
        text: The user message text to classify.
        custom_rules: Optional custom rules to evaluate before defaults.

    Returns:
        ClassificationResult with category, confidence, and matched pattern.
    """
    if not text or not text.strip():
        return ClassificationResult(
            category=TaskCategory.HEARTBEAT,
            confidence=0.99,
            matched_pattern="empty message",
        )

    stripped = text.strip()

    # Evaluate custom rules first (if provided)
    if custom_rules:
        for rule in custom_rules:
            if rule.pattern.search(stripped):
                return ClassificationResult(
                    category=rule.category,
                    confidence=rule.confidence,
                    matched_pattern=rule.description,
                )

    # Evaluate default patterns
    for rule in DEFAULT_PATTERNS:
        if rule.pattern.search(stripped):
            return ClassificationResult(
                category=rule.category,
                confidence=rule.confidence,
                matched_pattern=rule.description,
            )

    # Default fallback: longer messages get mid tier (simple-chat)
    return ClassificationResult(
        category=TaskCategory.SIMPLE_CHAT,
        confidence=0.40,
        matched_pattern="no pattern matched (default fallback)",
    )
