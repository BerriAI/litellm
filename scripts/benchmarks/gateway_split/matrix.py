import itertools
from typing import Final, Literal

from bifrost_bench import FrozenModel
from pydantic import Field, computed_field

TOPOLOGIES: Final = ("direct_axum", "direct_fastapi", "axum_python", "fastapi_rust")


PROFILE_NAMES: Final = ("litellm-current",)


class Job(FrozenModel):
    profile: Literal["litellm-current"]
    prompt_bytes: Literal[262144, 1048576, 4194304]
    response: Literal["buffered", "sse"]
    concurrency: Literal[1, 1000, 2000]
    seconds: int = Field(default=60, ge=1)
    repetitions: int = Field(default=3, ge=1)
    warmup_seconds: int = Field(default=5, ge=1)
    seed: int = 20261019

    @computed_field
    @property
    def id(self) -> str:
        return f"{self.profile}-{self.prompt_bytes}-{self.response}-c{self.concurrency}"

    @property
    def expected_rows(self) -> int:
        return len(TOPOLOGIES) * self.repetitions


def jobs(seconds: int = 60, repetitions: int = 3, warmup_seconds: int = 5) -> tuple[Job, ...]:
    combinations: Final = itertools.product(
        PROFILE_NAMES, (262144, 1048576, 4194304), ("buffered", "sse"), (1, 1000, 2000)
    )
    return tuple(
        Job.model_validate(
            {
                "profile": profile,
                "prompt_bytes": size,
                "response": response,
                "concurrency": concurrency,
                "seconds": seconds,
                "repetitions": repetitions,
                "warmup_seconds": warmup_seconds,
            }
        )
        for profile, size, response, concurrency in combinations
    )


def shard(selected: tuple[Job, ...], index: int, count: int) -> tuple[Job, ...]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("Shard index must be within the positive shard count")
    return tuple(job for position, job in enumerate(selected) if position % count == index)
