import itertools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from functools import partial
from typing import Final

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import assert_never

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


@dataclass(frozen=True, slots=True)
class Recent:
    since: date
    until: date
    checked: frozenset[tuple[str, str]] = frozenset()


Selector = OnDay | Exact | Recent


@dataclass(frozen=True, slots=True)
class Handled:
    finding: Finding
    by: str


@dataclass(frozen=True, slots=True)
class WatchDeps:
    fetch_text: Callable[[str], str]
    triage: Callable[[Candidate], TriageVerdict]
    already_handled: Callable[[Finding], str | None]
    implement: Callable[[Finding], Implementation]
    sdks: tuple[Sdk, ...]
    selector: Selector
    max_implementations: int
    carried: tuple[Finding, ...] = ()


@dataclass(frozen=True, slots=True)
class WatchState:
    candidates: tuple[Candidate, ...] = ()
    findings: tuple[Finding, ...] = ()
    fresh: tuple[Finding, ...] = ()
    handled: tuple[Handled, ...] = ()
    implementations: tuple[Implementation, ...] = ()
    deferred: tuple[Finding, ...] = ()


def select(selector: Selector, sdk: Sdk, releases: tuple[Release, ...]) -> tuple[Release, ...]:
    match selector:
        case OnDay(day=day):
            return tuple(release for release in releases if release.released == day)
        case Exact(sdk=name, version=version):
            return tuple(release for release in releases if sdk.name == name and release.version == version)
        case Recent(since=since, until=until, checked=checked):
            in_window: Final = (release for release in releases if since <= release.released < until)
            return tuple(release for release in in_window if (sdk.name, release.version) not in checked)
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
    triaged: Final = itertools.chain.from_iterable(_findings(deps, candidate) for candidate in state.candidates)
    findings: Final = itertools.chain(deps.carried, triaged)
    return {"findings": tuple({finding.branch: finding for finding in findings}.values())}


def _dedupe(deps: WatchDeps, state: WatchState) -> dict[str, tuple[Finding, ...] | tuple[Handled, ...]]:
    verdicts: Final = tuple((finding, deps.already_handled(finding)) for finding in state.findings)
    return {
        "fresh": tuple(finding for finding, by in verdicts if by is None),
        "handled": tuple(Handled(finding=finding, by=by) for finding, by in verdicts if by is not None),
    }


def _implement(deps: WatchDeps, state: WatchState) -> dict[str, tuple[Implementation, ...] | tuple[Finding, ...]]:
    return {
        "implementations": tuple(map(deps.implement, state.fresh[: deps.max_implementations])),
        "deferred": state.fresh[deps.max_implementations :],
    }


def _route(has_work: Callable[[WatchState], bool], node: str, state: WatchState) -> str:
    return node if has_work(state) else END


def build_graph(deps: WatchDeps) -> CompiledStateGraph:
    graph: Final = StateGraph(WatchState)
    graph.add_node("fetch", partial(_fetch, deps))
    graph.add_node("triage", partial(_triage, deps))
    graph.add_node("dedupe", partial(_dedupe, deps))
    graph.add_node("implement", partial(_implement, deps))
    graph.add_edge(START, "fetch")
    graph.add_conditional_edges(
        "fetch", partial(_route, lambda state: bool(state.candidates or deps.carried), "triage")
    )
    graph.add_conditional_edges("triage", partial(_route, lambda state: bool(state.findings), "dedupe"))
    graph.add_conditional_edges("dedupe", partial(_route, lambda state: bool(state.fresh), "implement"))
    graph.add_edge("implement", END)
    return graph.compile()


def run_watch(deps: WatchDeps) -> WatchState:
    return WatchState(**build_graph(deps).invoke(WatchState()))
