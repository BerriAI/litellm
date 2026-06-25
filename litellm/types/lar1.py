from enum import Enum
from typing import List

from pydantic import BaseModel


class LAR1Act(str, Enum):
    INF = "INF"  # Inference
    OBS = "OBS"  # Observation
    RET = "RET"  # Retrieval
    GEN = "GEN"  # Generation


class LAR1Time(str, Enum):
    NOW = "NOW"  # Current context
    MEM = "MEM"  # From memory
    CTX = "CTX"  # From context window
    PRE = "PRE"  # Predicted/future


class LAR1Mind(str, Enum):
    REF = "REF"  # Reflective
    REC = "REC"  # Recognized pattern
    HYP = "HYP"  # Hypothesis
    ACT = "ACT"  # Recommended action


class LAR1Metadata(BaseModel):
    act: LAR1Act = LAR1Act.INF
    time: LAR1Time = LAR1Time.NOW
    mind: LAR1Mind = LAR1Mind.REF
    confidence: float = 0.5  # 0.0-1.0
    evidence: List[str] = []  # "SYNTH", "RETRIEVED", "UNVERIFIED"
