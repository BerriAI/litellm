from dataclasses import dataclass
from importlib.resources import files
from typing import Final


def load(name: str) -> str:
    return files(__name__).joinpath(f"{name}.md").read_text().strip().replace("\n", " ")


@dataclass(frozen=True, slots=True)
class Prompts:
    review: str
    cluster: str
    investigate: str


PROMPTS: Final = Prompts(review=load("review"), cluster=load("cluster"), investigate=load("investigate"))
