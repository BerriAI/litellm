from collections.abc import Mapping
from datetime import date, datetime
from itertools import chain
from types import MappingProxyType
from typing import Final

import httpx

from litellm.proxy.roi_calculator.github import SourceError
from litellm.proxy.roi_calculator.observed_analytics import reporting_windows, summarize_observed
from litellm.proxy.roi_calculator.observed_sync import Progress, collect_observed
from litellm.proxy.roi_calculator.settings import StoredConnection, connection_id
from litellm.proxy.roi_calculator.source import repository_tag
from litellm.proxy.roi_calculator.sync import BranchSpendReader, GatewayUserReader, SpendReader
from litellm.types.roi_calculator import ROISettings, ROISpendRecord
from litellm.types.roi_observed import ObservedData, ObservedPeriodData, ObservedReport, ObservedSource


def source_details(settings: ROISettings) -> ObservedSource:
    return ObservedSource(
        id=connection_id(settings.source_provider, settings.source_api_url),
        source_provider=settings.source_provider,
        api_url=settings.source_api_url,
        repos=settings.repos,
    )


def scoped_data(data: ObservedData, source: ObservedSource) -> ObservedData:
    def period(value: ObservedPeriodData) -> ObservedPeriodData:
        return value.model_copy(
            update={"pulls": tuple(pull.model_copy(update={"connection_id": source.id}) for pull in value.pulls)}
        )

    return data.model_copy(
        update={
            "connections": (source,),
            "current": period(data.current),
            "previous": period(data.previous),
            "last_year": period(data.last_year),
        }
    )


def summarize_workspace(data: ObservedData, connections: tuple[StoredConnection, ...]) -> ObservedReport:
    included: Final = frozenset(source.id for source in data.connections)
    active: Final = tuple(entry for entry in connections if entry.id in included)
    maps: Final = ({f"{entry.id}:{login}": email for login, email in entry.identity_map.items()} for entry in active)
    identities: Final = MappingProxyType(dict(chain.from_iterable(mapping.items() for mapping in maps)))
    ignored: Final = tuple(
        chain.from_iterable(tuple(f"{entry.id}:{login}" for login in entry.ignored_logins) for entry in connections)
    )
    return summarize_observed(data, identities, ignored)


def combine_observed(sources: tuple[ObservedData, ...], repos: tuple[str, ...]) -> ObservedData:
    first: Final = sources[0]

    def period(values: tuple[ObservedPeriodData, ...]) -> ObservedPeriodData:
        if any(value.window != values[0].window or value.spend != values[0].spend for value in values):
            raise SourceError("The reporting windows changed during sync. Retry to get a complete report.")
        pulls: Final = tuple(chain.from_iterable(value.pulls for value in values))
        if len({pull.url for pull in pulls}) != len(pulls):
            raise SourceError("A repository is selected through more than one connection. Select it once.")
        return ObservedPeriodData(
            window=values[0].window,
            pulls=tuple(sorted(pulls, key=lambda pull: (pull.merged_at, pull.url), reverse=True)),
            issues=None
            if all(value.issues is None for value in values)
            else tuple(chain.from_iterable(value.issues or () for value in values)),
            spend=values[0].spend,
            branch_spend=None
            if any(value.branch_spend is None for value in values)
            else tuple(
                {
                    (row.repo, row.branch): row
                    for row in chain.from_iterable(value.branch_spend or () for value in values)
                }.values()
            ),
        )

    providers: Final = frozenset(source.source_provider for source in sources)
    return ObservedData(
        source_provider=first.source_provider if len(providers) == 1 else "mixed",
        source_api_url=first.source_api_url if len(sources) == 1 else "",
        connections=tuple(chain.from_iterable(source.connections for source in sources)),
        repos=repos,
        captured_at=first.captured_at,
        gateway_emails=first.gateway_emails,
        current=period(tuple(source.current for source in sources)),
        previous=period(tuple(source.previous for source in sources)),
        last_year=period(tuple(source.last_year for source in sources)),
    )


async def collect_workspace(
    connections: tuple[tuple[ROISettings, BranchSpendReader], ...],
    spend_reader: SpendReader,
    gateway_user_reader: GatewayUserReader,
    now: datetime,
    progress: Progress,
    transport: httpx.AsyncBaseTransport | None = None,
    days: int = 28,
) -> ObservedData:
    windows: Final = reporting_windows(now, days)
    spending: Final[Mapping[tuple[date, date], tuple[ROISpendRecord, ...]]] = {
        (window.start, window.end): await spend_reader(window.start, window.end) for window in windows
    }
    emails: Final = await gateway_user_reader()
    total: Final = sum(len(settings.repos) * 3 for settings, _reader in connections)

    async def spend(start: date, end: date) -> tuple[ROISpendRecord, ...]:
        return spending[(start, end)]

    async def users() -> frozenset[str]:
        return emails

    async def collect(index: int, settings: ROISettings, branch_reader: BranchSpendReader) -> ObservedData:
        offset: Final = sum(len(prior.repos) * 3 for prior, _reader in connections[:index])

        def update(stage: str, done: int, _total: int) -> None:
            progress(stage, offset + done, total)

        data: Final = await collect_observed(settings, spend, users, branch_reader, now, update, transport, days=days)
        return scoped_data(data, source_details(settings))

    data: Final = tuple(
        [await collect(index, settings, reader) for index, (settings, reader) in enumerate(connections)]
    )
    repos: Final = tuple(
        chain.from_iterable(
            tuple(repository_tag(settings, repo) if len(connections) > 1 else repo for repo in settings.repos)
            for settings, _reader in connections
        )
    )
    return combine_observed(data, repos)
