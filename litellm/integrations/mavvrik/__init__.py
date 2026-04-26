"""Mavvrik cost-data integration for LiteLLM.

Module layout:
  exporter.py      — Exporter (DB queries + DataFrame → CSV transform)
  uploader.py      — Uploader (GCS resumable upload protocol)
  client.py        — Client (Mavvrik REST API calls + retry transport)
  settings.py      — Settings (config detection and persistence)
  orchestrator.py  — Orchestrator (pod lock + register → date loop → upload → advance)

Public facade:
  Service — used by mavvrik_endpoints.py; all business logic lives here.
"""

import os
from datetime import datetime, timedelta
from datetime import timezone as _tz
from typing import Optional

from litellm._logging import verbose_proxy_logger
from litellm.constants import MAVVRIK_MAX_FETCHED_DATA_RECORDS
from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.exporter import Exporter
from litellm.integrations.mavvrik.logger import Logger
from litellm.integrations.mavvrik.orchestrator import Orchestrator
from litellm.integrations.mavvrik.settings import Settings
from litellm.integrations.mavvrik.uploader import Uploader

__all__ = [
    "Client",
    "Exporter",
    "Logger",
    "Orchestrator",
    "Service",
    "Settings",
    "Uploader",
]


def _build_client(data: dict) -> Client:
    """Build a Client from loaded settings dict."""
    return Client(
        api_key=data.get("api_key") or os.getenv("MAVVRIK_API_KEY", ""),
        api_endpoint=data.get("api_endpoint") or os.getenv("MAVVRIK_API_ENDPOINT", ""),
        connection_id=data.get("connection_id")
        or os.getenv("MAVVRIK_CONNECTION_ID", ""),
    )


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

        client = Client(
            api_key=api_key,
            api_endpoint=api_endpoint,
            connection_id=connection_id,
        )
        uploader = Uploader(client=client)
        orchestrator = Orchestrator(client=client, uploader=uploader)
        # replace_existing=True ensures repeated /mavvrik/init calls are safe.
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

        Falls back to env vars when no DB settings exist.

        Returns:
            A dict with keys: api_key_masked, api_endpoint, connection_id, status.
        """
        from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

        data = await self._settings.load()

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

        Raises:
            LookupError: when no existing settings are found.
            ValueError: when a merge would leave a required field empty.
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
        """
        from litellm.constants import MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME

        import litellm.proxy.proxy_server as _pserver

        await self._settings.delete()

        _scheduler = getattr(_pserver, "scheduler", None)
        if _scheduler is not None:
            try:
                _scheduler.remove_job(MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME)
            except Exception:
                pass  # job may not exist if scheduler was restarted

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
        """Fetch spend data and upload to Mavvrik for a calendar date.

        Args:
            date_str: YYYY-MM-DD.  Defaults to yesterday (UTC) when omitted.
            limit:    Cap the number of rows fetched from the database.

        Raises:
            ValueError: when Mavvrik is not configured.

        Returns:
            {"message": str, "status": "success", "records_exported": int}
        """
        data = await self._settings.load()

        if not data and not self._settings.has_env_vars:
            raise ValueError("Mavvrik not configured. Call POST /mavvrik/init first.")

        self._settings._ensure_prisma_client()

        date_str = date_str or self._yesterday()
        effective_limit = limit or MAVVRIK_MAX_FETCHED_DATA_RECORDS

        client = _build_client(data)
        uploader = Uploader(client=client)
        exporter = Exporter()

        df, csv_payload = await exporter.export(
            date_str=date_str,
            connection_id=client.connection_id,
            limit=effective_limit,
        )

        if df.is_empty():
            return {
                "message": f"No data for {date_str}",
                "status": "success",
                "records_exported": 0,
            }

        records_exported = len(df)
        await uploader.upload(csv_payload, date_str=date_str)

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
        data = await self._settings.load()
        if not data and not self._settings.has_env_vars:
            raise ValueError("Mavvrik not configured. Call POST /mavvrik/init first.")

        self._settings._ensure_prisma_client()

        date_str = date_str or self._yesterday()
        effective_limit = limit or MAVVRIK_MAX_FETCHED_DATA_RECORDS

        client = _build_client(data)
        exporter = Exporter()

        df, csv_payload = await exporter.export(
            date_str=date_str,
            connection_id=client.connection_id,
            limit=effective_limit,
        )

        if df.is_empty():
            return {
                "message": "Mavvrik dry run completed",
                "status": "success",
                "dry_run_data": {"usage_data": [], "csv_preview": ""},
                "summary": {
                    "total_records": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "unique_models": 0,
                    "unique_teams": 0,
                },
            }

        total_cost = float(df["spend"].sum()) if "spend" in df.columns else 0.0
        total_tokens = (
            int((df["prompt_tokens"].sum() or 0) + (df["completion_tokens"].sum() or 0))
            if "prompt_tokens" in df.columns
            else 0
        )
        unique_models = df["model"].n_unique() if "model" in df.columns else 0
        unique_teams = df["team_id"].n_unique() if "team_id" in df.columns else 0

        return {
            "message": "Mavvrik dry run completed",
            "status": "success",
            "dry_run_data": {
                "usage_data": df.head(50).to_dicts(),
                "csv_preview": csv_payload[:5000] if csv_payload else "",
            },
            "summary": {
                "total_records": len(df),
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "unique_models": unique_models,
                "unique_teams": unique_teams,
            },
        }
