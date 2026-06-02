"""
Input guardrails — first line of defence before query reaches retrieval.

Checks (in order):
1. Length limits
2. SQL injection patterns
3. Prompt injection patterns
4. PII fishing patterns (asking system to reveal personal data)
5. Confidential data extraction patterns
6. Off-topic / dangerous content patterns

Returns a GuardResult — either PASS or BLOCK with a reason.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    PASS  = "pass"
    BLOCK = "block"


@dataclass
class GuardResult:
    decision: Decision
    reason:   str = ""

    @property
    def blocked(self) -> bool:
        return self.decision == Decision.BLOCK


# ── Pattern lists ─────────────────────────────────────────────────────────────

_SQL_PATTERNS = [
    r"(--|;|/\*|\*/)",                          # SQL comment / terminator
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|UNION|CAST)\b",
    r"\b(OR|AND)\s+[\w'\"]+\s*=\s*[\w'\"]+",   # OR 1=1 style
    r"xp_\w+",                                  # SQL Server stored procs
    r"SLEEP\s*\(",                              # time-based blind injection
    r"BENCHMARK\s*\(",
]

_PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(your|the)\s+(system\s+)?prompt",
    r"you\s+are\s+now\s+(a\s+)?(\w+\s+)?assistant",
    r"new\s+(role|persona|instruction)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+you\s+are\s+)?a?n?\s+\w+\s*(with\s+no\s+restrictions)?",
    r"do\s+anything\s+now",                    # DAN jailbreak
    r"jailbreak",
    r"override\s+(safety|guidelines|restrictions)",
    r"system\s*:\s*you\s+are",                 # fake system prompt
]

_PII_FISHING_PATTERNS = [
    r"(show|give|list|reveal|expose|extract)\s+(me\s+)?(all\s+)?(user|customer|patient|employee|personal)\s+(data|info|details|records|passwords?|emails?|phones?|addresses?)",
    r"(what\s+is|tell\s+me)\s+(the\s+)?(password|email|phone|address|ssn|social\s+security)",
    r"(dump|export|download)\s+(all\s+)?(the\s+)?(database|data|records|users)",
    r"(who\s+are\s+all\s+the|list\s+all)\s+(users|customers|clients|employees)",
    r"(show|give|list|reveal)\s+(me\s+)?(all\s+)?(passwords?|emails?|phone\s+numbers?|pii)",
]

_CONFIDENTIAL_PATTERNS = [
    r"(reveal|show|expose|leak)\s+(\w+\s+)?(confidential|secret|private|internal|proprietary)",
    r"(reveal|show|expose|leak)\s+(your|the|my|our)\s+(\w+\s+)?(api.?key|credential|secret|token|password)",
    r"(what\s+are\s+your|show\s+me\s+your)\s+(api\s+key|credentials|secrets|tokens|passwords)",
    r"(access|read|view)\s+(system\s+)?(config|configuration|env|environment\s+variable)",
    r"internal\s+(document|memo|report|strategy)",
]

_DANGEROUS_PATTERNS = [
    r"\b(hack|exploit|vulnerability|bypass|crack|brute.?force)\b",
    r"(how\s+to|steps\s+to)\s+(make|create|build)\s+(weapon|bomb|malware|virus|ransomware)",
]


def _matches_any(text: str, patterns: list[str]) -> str | None:
    """Return the first matching pattern string, or None."""
    t = text.upper() if any(k in p for p in patterns for k in ["SELECT", "DROP", "UNION"]) else text
    for pat in patterns:
        if re.search(pat, t, re.IGNORECASE):
            return pat
    return None


# ── Public API ────────────────────────────────────────────────────────────────

MAX_QUESTION_LEN = 500
MIN_QUESTION_LEN = 3


def check(question: str) -> GuardResult:
    """Run all input checks. Returns GuardResult(PASS) or GuardResult(BLOCK, reason)."""

    # 1 — Length
    if len(question) < MIN_QUESTION_LEN:
        return GuardResult(Decision.BLOCK, "Query too short.")
    if len(question) > MAX_QUESTION_LEN:
        return GuardResult(Decision.BLOCK, f"Query exceeds {MAX_QUESTION_LEN} characters.")

    # 2 — SQL injection
    if _matches_any(question, _SQL_PATTERNS):
        return GuardResult(Decision.BLOCK, "Query contains disallowed patterns.")

    # 3 — Prompt injection
    if _matches_any(question, _PROMPT_INJECTION_PATTERNS):
        return GuardResult(Decision.BLOCK, "Query contains disallowed instruction patterns.")

    # 4 — PII fishing
    if _matches_any(question, _PII_FISHING_PATTERNS):
        return GuardResult(Decision.BLOCK, "Requests for personal data are not permitted.")

    # 5 — Confidential data extraction
    if _matches_any(question, _CONFIDENTIAL_PATTERNS):
        return GuardResult(Decision.BLOCK, "Requests for confidential system information are not permitted.")

    # 6 — Dangerous content
    if _matches_any(question, _DANGEROUS_PATTERNS):
        return GuardResult(Decision.BLOCK, "Query contains disallowed content.")

    return GuardResult(Decision.PASS)
