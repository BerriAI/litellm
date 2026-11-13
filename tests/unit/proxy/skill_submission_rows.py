from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

from prisma import models as prisma_models


class SubmissionRows:
    def __init__(self) -> None:
        self.rows: tuple[prisma_models.LiteLLM_SkillSubmission, ...] = ()

    async def create(self, data: Mapping[str, object]) -> prisma_models.LiteLLM_SkillSubmission:
        row: Final = prisma_models.LiteLLM_SkillSubmission.model_validate(
            {
                "id": f"submission-{len(self.rows)}",
                "status": "pending_review",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                **data,
            }
        )
        self.rows = (*self.rows, row)
        return row

    async def find_unique(self, where: Mapping[str, object]) -> prisma_models.LiteLLM_SkillSubmission | None:
        return next((row for row in self.rows if row.id == where["id"]), None)

    async def find_first(self, where: Mapping[str, object]) -> prisma_models.LiteLLM_SkillSubmission | None:
        row: Final = await self.find_unique(where)
        return None if row is None or where.get("submitted_by", row.submitted_by) != row.submitted_by else row

    async def update(
        self,
        where: Mapping[str, object],
        data: Mapping[str, object],
    ) -> prisma_models.LiteLLM_SkillSubmission | None:
        current: Final = await self.find_unique(where)
        if current is None:
            return None
        updated: Final = prisma_models.LiteLLM_SkillSubmission.model_validate({**current.model_dump(), **data})
        self.rows = tuple(updated if row.id == current.id else row for row in self.rows)
        return updated

    async def query_raw(self, query: str, *args: object) -> tuple[Mapping[str, object], ...]:
        if query.startswith("SELECT scan_json"):
            return tuple({"scan_json": row.scan_json} for row in self.rows if row.id == args[0])
        if 'FROM "LiteLLM_SkillSubmission"' not in query or "FOR UPDATE" in query:
            return ()
        if "COUNT(*) AS count" in query:
            pending: Final = tuple(
                row for row in self.rows if row.submitted_by == args[0] and row.status == "pending_review"
            )
            return (
                {
                    "count": len(pending),
                    "storage_bytes": sum(
                        len(row.proposal_json.encode("utf-8"))
                        + (0 if row.archive_content is None else len(row.archive_content.decode()))
                        for row in pending
                    ),
                },
            )
        records: Final = (
            tuple(row for row in self.rows if row.id == args[0] and (args[1] is None or row.submitted_by == args[1]))
            if "WHERE id = $1" in query
            else tuple(row for row in self.rows if args[0] is None or row.submitted_by == args[0])
        )
        if "WHERE id = $1" in query:
            return tuple({**row.model_dump(), "has_archive": row.archive_content is not None} for row in records)
        limit, offset = args[2:]
        assert isinstance(limit, int) and isinstance(offset, int)
        filtered: Final = tuple(row for row in records if args[1] is None or row.status == args[1])
        ordered: Final = sorted(filtered, key=lambda row: (row.created_at, row.id), reverse=True)
        return tuple(
            {**row.model_dump(), "has_archive": row.archive_content is not None}
            for row in ordered[offset : offset + limit]
        )
