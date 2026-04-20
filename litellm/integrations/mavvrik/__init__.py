"""Mavvrik cost-data integration for LiteLLM.

Module layout:
  exporter.py      — MavvrikExporter (CustomLogger subclass, core export pipeline)
  client.py        — MavvrikClient (3-step signed URL upload + register/advance_marker)
  database.py      — MavvrikDatabase (DB queries)
  transform.py     — MavvrikTransformer (DataFrame → CSV)
  settings.py      — MavvrikSettings (config detection and persistence)
  orchestrator.py  — MavvrikOrchestrator (register → date loop → export → advance)
  scheduler.py     — MavvrikScheduler (APScheduler registration + pod lock)

Public facade:
  MavvrikService — used by mavvrik_endpoints.py; all business logic lives here.
"""

from datetime import datetime, timedelta
from datetime import timezone as _tz
from typing import Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.mavvrik.settings import MavvrikSettings
from litellm.integrations.mavvrik.scheduler import MavvrikScheduler


class MavvrikService:
    """Public facade that mediates between the REST endpoints and the Mavvrik modules.

    Each method maps 1-to-1 with an endpoint action.  All methods return plain
    ``dict`` objects so the router can freely convert them into response models.

    Raises:
        LookupError  — resource not found (router → 404)
        ValueError   — bad input (router → 400)
        RuntimeError — upstream / integration failure (router → 500)
        Exception    — catch-all (router → 500)
    """

    def __init__(self) -> None:
        self._settings = MavvrikSettings()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def settings(self) -> MavvrikSettings:
        return self._settings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _yesterday() -> str:
        """Return yesterday's date as YYYY-MM-DD (UTC)."""
        return (datetime.now(_tz.utc).date() - timedelta(days=1)).isoformat()

    # ------------------------------------------------------------------
    # initialize  →  POST /mavvrik/init
    # ------------------------------------------------------------------

    async def initialize(
        self,
        api_key: str,
        api_endpoint: str,
        connection_id: str,
    ) -> dict:
        """Save credentials and schedule the export job.

        The export marker (cursor) is owned exclusively by the Mavvrik API —
        it is retrieved via register() at the start of each scheduled run,
        not stored locally.

        Returns:
            {"message": str, "status": "success"}
        """
        # Step 1 — persist credentials (no marker stored locally).
        await self._settings.save(
            api_key=api_key,
            api_endpoint=api_endpoint,
            connection_id=connection_id,
        )

        # Step 3 — schedule the background export job.
        try:
            import litellm.proxy.proxy_server as _pserver

            _scheduler = getattr(_pserver, "scheduler", None)
            await MavvrikScheduler.register_exporter_and_job(
                api_key=api_key,
                api_endpoint=api_endpoint,
                connection_id=connection_id,
                scheduler=_scheduler,
            )
        except Exception as sched_exc:
            verbose_proxy_logger.warning(
                "Mavvrik: could not register background job after init (%s)", sched_exc
            )

        return {
            "message": "Mavvrik settings initialized successfully",
            "status": "success",
        }

    # ------------------------------------------------------------------
    # get_settings  →  GET /mavvrik/settings
    # ------------------------------------------------------------------

    async def get_settings(self) -> dict:
        """Load and mask Mavvrik settings.

        Returns:
            A dict with keys: api_key_masked, api_endpoint, connection_id, status.
            ``status`` is ``"not_configured"`` when no settings exist, otherwise ``"configured"``.
            The export marker is owned by the Mavvrik API and is not stored locally.
        """
        from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

        data = await self._settings.load()
        if not data:
            return {
                "api_key_masked": None,
                "api_endpoint": None,
                "connection_id": None,
                "status": "not_configured",
            }

        masker = SensitiveDataMasker()
        masked = masker.mask_dict({"api_key": data.get("api_key", "")})
        return {
            "api_key_masked": masked.get("api_key"),
            "api_endpoint": data.get("api_endpoint"),
            "connection_id": data.get("connection_id"),
            "status": "configured",
        }

    # ------------------------------------------------------------------
    # update_settings  →  PUT /mavvrik/settings
    # ------------------------------------------------------------------

    async def update_settings(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        connection_id: Optional[str] = None,
    ) -> dict:
        """Merge new credential values into existing settings and persist.

        The export marker is owned by the Mavvrik API and cannot be set here.

        Returns:
            {"message": str, "status": "success"}
        """
        current = await self._settings.load()

        def _pick(new: Optional[str], key: str) -> str:
            return new if new is not None else current.get(key, "")

        await self._settings.save(
            api_key=_pick(api_key, "api_key"),
            api_endpoint=_pick(api_endpoint, "api_endpoint"),
            connection_id=_pick(connection_id, "connection_id"),
        )
        return {"message": "Mavvrik settings updated successfully", "status": "success"}

    # ------------------------------------------------------------------
    # delete  →  DELETE /mavvrik/delete
    # ------------------------------------------------------------------

    async def delete(self) -> dict:
        """Remove all Mavvrik settings and deregister the scheduler job.

        Raises:
            LookupError: when no settings exist in the database.

        Returns:
            {"message": str, "status": "success"}
        """
        from litellm.constants import MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME

        # Raises LookupError if not configured.
        await self._settings.delete()

        # Best-effort scheduler cleanup.
        try:
            import litellm.proxy.proxy_server as _pserver

            _scheduler = getattr(_pserver, "scheduler", None)
            if _scheduler is not None:
                try:
                    _scheduler.remove_job(MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME)
                except Exception as exc:
                    verbose_proxy_logger.debug(
                        "Mavvrik: scheduler job already removed or not found: %s", exc
                    )
        except Exception as exc:
            verbose_proxy_logger.debug(
                "Mavvrik: could not access scheduler during delete: %s", exc
            )

        verbose_proxy_logger.info("Mavvrik settings deleted")
        return {"message": "Mavvrik settings deleted successfully", "status": "success"}

    # ------------------------------------------------------------------
    # export  →  POST /mavvrik/export
    # ------------------------------------------------------------------

    async def export(
        self,
        date_str: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Export spend data for a calendar date to Mavvrik.

        Args:
            date_str: YYYY-MM-DD.  Defaults to yesterday (UTC) when omitted.
            limit:    Cap the number of rows fetched from the database.

        Raises:
            ValueError: when Mavvrik is not configured.

        Returns:
            {"message": str, "status": "success", "records_exported": int}
        """
        from litellm.integrations.mavvrik.exporter import MavvrikExporter

        data = await self._settings.load()
        if not data:
            raise ValueError("Mavvrik not configured. Call POST /mavvrik/init first.")

        date_str = date_str or self._yesterday()

        exporter = MavvrikExporter(
            api_key=data.get("api_key"),
            api_endpoint=data.get("api_endpoint"),
            connection_id=data.get("connection_id"),
        )
        records_exported = await exporter.export_usage_data(
            date_str=date_str,
            limit=limit,
        )
        return {
            "message": f"Mavvrik export completed successfully for {date_str}",
            "status": "success",
            "records_exported": records_exported,
        }

    # ------------------------------------------------------------------
    # dry_run  →  POST /mavvrik/dry-run
    # ------------------------------------------------------------------

    async def dry_run(
        self,
        date_str: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Preview CSV records without uploading.

        Args:
            date_str: YYYY-MM-DD.  Defaults to yesterday (UTC) when omitted.
            limit:    Cap the number of rows fetched from the database.

        Returns:
            {"message": str, "status": "success", "dry_run_data": dict, "summary": dict}
        """
        from litellm.integrations.mavvrik.exporter import MavvrikExporter

        data = await self._settings.load()
        date_str = date_str or self._yesterday()

        exporter = MavvrikExporter(
            api_key=data.get("api_key"),
            api_endpoint=data.get("api_endpoint"),
            connection_id=data.get("connection_id"),
        )
        result = await exporter.dry_run(date_str=date_str, limit=limit)
        return {
            "message": "Mavvrik dry run completed",
            "status": "success",
            "dry_run_data": {
                "usage_data": result["usage_data"],
                "csv_preview": result["csv_preview"],
            },
            "summary": result["summary"],
        }
