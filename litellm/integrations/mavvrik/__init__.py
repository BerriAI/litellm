"""Mavvrik cost-data integration for LiteLLM.

Module layout:
  exporter.py      — Exporter (DB queries + DataFrame → CSV transform)
  uploader.py      — Uploader (export → upload pipeline)
  client.py        — Client (3-step signed URL upload + register/advance_marker)
  settings.py      — Settings (config detection and persistence)
  orchestrator.py  — Orchestrator (pod lock + register → date loop → upload → advance)

Public facade:
  Service — used by mavvrik_endpoints.py; all business logic lives here.
"""

from datetime import datetime, timedelta
from datetime import timezone as _tz
from typing import Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.mavvrik.logger import Logger
from litellm.integrations.mavvrik.orchestrator import Orchestrator
from litellm.integrations.mavvrik.settings import Settings
from litellm.integrations.mavvrik.uploader import Uploader

__all__ = ["Logger", "Orchestrator", "Service", "Settings", "Uploader"]


class Service:
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
        self._settings = Settings()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
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
        # Step 1 — persist credentials.
        await self._settings.save(
            api_key=api_key,
            api_endpoint=api_endpoint,
            connection_id=connection_id,
        )

        # Step 2 — schedule the background export job.
        from litellm.constants import (
            MAVVRIK_EXPORT_INTERVAL_MINUTES,
            MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
        )
        from litellm.integrations.mavvrik.orchestrator import Orchestrator
        from litellm.integrations.mavvrik.uploader import Uploader

        import litellm.proxy.proxy_server as _pserver

        _scheduler = getattr(_pserver, "scheduler", None)
        if _scheduler is None:
            verbose_proxy_logger.warning(
                "mavvrik: scheduler not available, background job not registered"
            )
            return {
                "message": "Mavvrik settings initialized successfully",
                "status": "success",
            }

        uploader = Uploader(
            api_key=api_key,
            api_endpoint=api_endpoint,
            connection_id=connection_id,
        )
        orchestrator = Orchestrator(uploader=uploader)
        _scheduler.add_job(
            orchestrator.run,
            "interval",
            minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES,
            id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
            replace_existing=True,
        )
        verbose_proxy_logger.info(
            "mavvrik background export job scheduled every %d min",
            MAVVRIK_EXPORT_INTERVAL_MINUTES,
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

        Falls back to env vars when no DB settings exist, so operators using
        the env-var-only setup see ``status: "configured"`` rather than
        ``status: "not_configured"`` while exports are running.

        Returns:
            A dict with keys: api_key_masked, api_endpoint, connection_id, status.
            ``status`` is ``"not_configured"`` when neither DB nor env vars are set.
            The export marker is owned by the Mavvrik API and is not stored locally.
        """
        import os

        from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

        data = await self._settings.load()

        # Fall back to env vars for operators who never called /mavvrik/init.
        if not data and self._settings.has_env_vars:
            data = {
                "api_key": os.getenv("MAVVRIK_API_KEY", ""),
                "api_endpoint": os.getenv("MAVVRIK_API_ENDPOINT", ""),
                "connection_id": os.getenv("MAVVRIK_CONNECTION_ID", ""),
            }

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

        Raises:
            LookupError: when no existing settings are found (use POST /mavvrik/init instead).
            ValueError: when a merge would leave a required field empty.

        Returns:
            {"message": str, "status": "success"}
        """
        current = await self._settings.load()
        if not current:
            raise LookupError(
                "Mavvrik settings not found. Use POST /mavvrik/init to create them first."
            )

        def _pick(new: Optional[str], key: str) -> str:
            return new if new is not None else current.get(key, "")

        merged = {
            "api_key": _pick(api_key, "api_key"),
            "api_endpoint": _pick(api_endpoint, "api_endpoint"),
            "connection_id": _pick(connection_id, "connection_id"),
        }
        missing = [k for k, v in merged.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required Mavvrik settings after merge: {missing}"
            )

        await self._settings.save(**merged)
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

        import litellm.proxy.proxy_server as _pserver

        # Raises LookupError if not configured.
        await self._settings.delete()

        _scheduler = getattr(_pserver, "scheduler", None)
        if _scheduler is not None:
            _scheduler.remove_job(MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME)

        verbose_proxy_logger.info("mavvrik settings deleted")
        return {"message": "Mavvrik settings deleted successfully", "status": "success"}

    # ------------------------------------------------------------------
    # export  →  POST /mavvrik/export
    # ------------------------------------------------------------------

    async def export(
        self,
        date_str: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Upload spend data for a calendar date to Mavvrik.

        Args:
            date_str: YYYY-MM-DD.  Defaults to yesterday (UTC) when omitted.
            limit:    Cap the number of rows fetched from the database.

        Raises:
            ValueError: when Mavvrik is not configured.

        Returns:
            {"message": str, "status": "success", "records_exported": int}
        """
        from litellm.integrations.mavvrik.uploader import Uploader

        data = await self._settings.load()

        # Allow export when credentials come from env vars (zero-config setup).
        if not data and not self._settings.has_env_vars:
            raise ValueError("Mavvrik not configured. Call POST /mavvrik/init first.")

        date_str = date_str or self._yesterday()

        uploader = Uploader(
            api_key=data.get("api_key") if data else None,
            api_endpoint=data.get("api_endpoint") if data else None,
            connection_id=data.get("connection_id") if data else None,
        )
        records_exported = await uploader.upload_usage_data(
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
        from litellm.integrations.mavvrik.uploader import Uploader

        data = await self._settings.load()
        if not data and not self._settings.has_env_vars:
            raise ValueError("Mavvrik not configured. Call POST /mavvrik/init first.")

        date_str = date_str or self._yesterday()

        uploader = Uploader(
            api_key=data.get("api_key") if data else None,
            api_endpoint=data.get("api_endpoint") if data else None,
            connection_id=data.get("connection_id") if data else None,
        )
        result = await uploader.dry_run(date_str=date_str, limit=limit)
        return {
            "message": "Mavvrik dry run completed",
            "status": "success",
            "dry_run_data": {
                "usage_data": result["usage_data"],
                "csv_preview": result["csv_preview"],
            },
            "summary": result["summary"],
        }


from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.exporter import Exporter
from litellm.integrations.mavvrik.orchestrator import Orchestrator
from litellm.integrations.mavvrik.settings import Settings
from litellm.integrations.mavvrik.uploader import Uploader

__all__ = ["Client", "Exporter", "Orchestrator", "Service", "Settings", "Uploader"]
