from enum import Enum

from pydantic import BaseModel, Field


class LAR1Act(str, Enum):
    INF = "INF"
    OBS = "OBS"
    RET = "RET"
    GEN = "GEN"


class LAR1Time(str, Enum):
    NOW = "NOW"
    MEM = "MEM"
    CTX = "CTX"
    PRE = "PRE"


class LAR1Mind(str, Enum):
    REF = "REF"
    REC = "REC"
    HYP = "HYP"
    ACT = "ACT"


class LAR1Evidence(str, Enum):
    SYNTH = "SYNTH"
    RETRIEVED = "RETRIEVED"
    UNVERIFIED = "UNVERIFIED"
    CONFIRMED = "CONFIRMED"


class LAR1Metadata(BaseModel):
    act: LAR1Act = LAR1Act.INF
    time: LAR1Time = LAR1Time.NOW
    mind: LAR1Mind = LAR1Mind.REF
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[LAR1Evidence] = []
