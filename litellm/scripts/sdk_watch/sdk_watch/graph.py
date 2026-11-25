import itertools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from functools import partial
from typing import Final, assert_never

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sdk_watch.changelog import Release, parse_changelog
from sdk_watch.implement import Implementation
from sdk_watch.sources import Candidate, Sdk
from sdk_watch.triage import Finding, TriageVerdict


@dataclass(frozen=True, slots=True)
class OnDay:
    day: date


@dataclass(frozen=True, slots=True)
class Exact:
    sdk: str
    version: str


Selector = OnDay | Exact


@dataclass(frozen=True, slots=True)
class WatchDeps:
    fetch_text: Callable[[str], str]
    triage: Callable[[Candidate], TriageVerdict]
    already_handled: Callable[[Finding], bool]
    implement: Callable[[Finding], Implementation]
    sdks: tuple[Sdk, ...]
    selector: Selector
    max_implementations: int


@dataclass(frozen=True, slots=True)
class WatchState:
    candidates: tuple[Candidate, ...] = ()
    findings: tuple[Finding, ...] = ()
    fresh: tuple[Finding, ...] = ()
    implementations: tuple[Implementation, ...] = ()


def select(selector: Selector, sdk: Sdk, releases: tuple[Release, ...]) -> tuple[Release, ...]:
    match selector:
        case OnDay(day=day):
            return tuple(release for release in releases if release.released == day)
        case Exact(sdk=name, version=version):
            return tuple(release for release in releases if sdk.name == name and release.version == version)
        case _:
            assert_never(selector)


def _candidates(deps: WatchDeps, sdk: Sdk) -> tuple[Candidate, ...]:
    releases: Final = select(deps.selector, sdk, parse_changelog(deps.fetch_text(sdk.changelog_url)))
    return tuple(Candidate(sdk=sdk, release=release) for release in releases)


def _findings(deps: WatchDeps, candidate: Candidate) -> tuple[Finding, ...]:
    return tuple(Finding(candidate=candidate, change=change) for change in deps.triage(candidate).changes)


def _fetch(deps: WatchDeps, state: WatchState) -> dict[str, tuple[Candidate, ...]]:
    return {"candidates": tuple(itertools.chain.from_iterable(_candidates(deps, sdk) for sdk in deps.sdks))}


def _triage(deps: WatchDeps, state: WatchState) -> dict[str, tuple[Finding, ...]]:
    findings: Final = itertools.chain.from_iterable(_findings(deps, candidate) for candidate in state.candidates)
    return {"findings": tuple({finding.branch: finding for finding in findings}.values())}


def _dedupe(deps: WatchDeps, state: WatchState) -> dict[str, tuple[Finding, ...]]:
    return {"fresh": tuple(finding for finding in state.findings if not deps.already_handled(finding))}


def _implement(deps: WatchDeps, state: WatchState) -> dict[str, tuple[Implementation, ...]]:
    return {"implementations": tuple(map(deps.implement, state.fresh[: deps.max_implementations]))}


def _route(has_work: Callable[[WatchState], bool], node: str, state: WatchState) -> str:
    return node if has_work(state) else END


def build_graph(deps: WatchDeps) -> CompiledStateGraph:
    graph: Final = StateGraph(WatchState)
    graph.add_node("fetch", partial(_fetch, deps))
    graph.add_node("triage", partial(_triage, deps))
    graph.add_node("dedupe", partial(_dedupe, deps))
    graph.add_node("implement", partial(_implement, deps))
    graph.add_edge(START, "fetch")
    graph.add_conditional_edges("fetch", partial(_route, lambda state: bool(state.candidates), "triage"))
    graph.add_conditional_edges("triage", partial(_route, lambda state: bool(state.findings), "dedupe"))
    graph.add_conditional_edges("dedupe", partial(_route, lambda state: bool(state.fresh), "implement"))
    graph.add_edge("implement", END)
    return graph.compile()


def run_watch(deps: WatchDeps) -> WatchState:
    return WatchState(**build_graph(deps).invoke(WatchState()))
