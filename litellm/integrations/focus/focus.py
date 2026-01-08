"""
FOCUS (FinOps Open Cost & Usage Specification) Exporter for LiteLLM.

This module provides functionality to export LiteLLM usage data in FOCUS format,
which is an open specification for consistent cost and usage datasets.

More info: https://focus.finops.org/
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import polars as pl

from litellm._logging import verbose_logger


class FOCUSExporter:
    """
    FOCUS Exporter for exporting LiteLLM usage data in FOCUS format.

    FOCUS (FinOps Open Cost & Usage Specification) is an open standard
    for cost and usage data interoperability supported by the FinOps Foundation.

    This exporter can output data in:
    - JSON format
    - CSV format
    - Parquet format

    Environment Variables:
        FOCUS_EXPORT_TIMEZONE: Timezone for date handling (default: UTC)
    """

    def __init__(
        self,
        timezone: Optional[str] = None,
        include_tags: bool = True,
        include_token_breakdown: bool = True,
        **kwargs,
    ):
        """
        Initialize FOCUS exporter with configuration.

        Args:
            timezone: Timezone for date handling (default: UTC)
            include_tags: Whether to include tags in output
            include_token_breakdown: Whether to include token breakdown in tags
        """
        self.timezone = timezone or os.getenv("FOCUS_EXPORT_TIMEZONE", "UTC")
        self.include_tags = include_tags
        self.include_token_breakdown = include_token_breakdown
        verbose_logger.debug(
            f"FOCUS Exporter initialized with timezone: {self.timezone}"
        )

    async def get_focus_data(
        self,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Get usage data from database and transform to FOCUS format.

        Args:
            limit: Optional limit on number of records
            start_time_utc: Optional start time filter
            end_time_utc: Optional end time filter

        Returns:
            Polars DataFrame with FOCUS formatted data
        """
        from litellm.integrations.focus.database import FOCUSDatabase
        from litellm.integrations.focus.transform import FOCUSTransformer

        try:
            verbose_logger.debug("FOCUS Exporter: Loading usage data from database")

            # Load data from database
            database = FOCUSDatabase()
            data = await database.get_usage_data(
                limit=limit,
                start_time_utc=start_time_utc,
                end_time_utc=end_time_utc,
            )

            if data.is_empty():
                verbose_logger.debug("FOCUS Exporter: No usage data found")
                return pl.DataFrame()

            verbose_logger.debug(f"FOCUS Exporter: Processing {len(data)} records")

            # Transform to FOCUS format
            transformer = FOCUSTransformer(
                include_tags=self.include_tags,
                include_token_breakdown=self.include_token_breakdown,
            )
            focus_data = transformer.transform(data)

            verbose_logger.debug(
                f"FOCUS Exporter: Transformed {len(focus_data)} records to FOCUS format"
            )

            return focus_data

        except Exception as e:
            verbose_logger.error(f"FOCUS Exporter: Error getting FOCUS data: {e}")
            raise

    async def export_json(
        self,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> str:
        """
        Export usage data to FOCUS format as JSON string.

        Args:
            limit: Optional limit on number of records
            start_time_utc: Optional start time filter
            end_time_utc: Optional end time filter

        Returns:
            JSON string with FOCUS formatted data
        """
        focus_data = await self.get_focus_data(
            limit=limit,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
        )

        if focus_data.is_empty():
            return json.dumps({"records": [], "record_count": 0})

        records = focus_data.to_dicts()
        
        # Convert datetime objects to ISO strings for JSON serialization
        for record in records:
            for key, value in record.items():
                if isinstance(value, datetime):
                    record[key] = value.isoformat()

        return json.dumps(
            {
                "focus_version": "1.0",
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "record_count": len(records),
                "records": records,
            },
            indent=2,
            default=str,
        )

    async def export_csv(
        self,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> str:
        """
        Export usage data to FOCUS format as CSV string.

        Args:
            limit: Optional limit on number of records
            start_time_utc: Optional start time filter
            end_time_utc: Optional end time filter

        Returns:
            CSV string with FOCUS formatted data
        """
        focus_data = await self.get_focus_data(
            limit=limit,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
        )

        if focus_data.is_empty():
            return ""

        # For CSV, convert Tags dict to JSON string
        if "Tags" in focus_data.columns:
            focus_data = focus_data.with_columns(
                pl.col("Tags").map_elements(
                    lambda x: json.dumps(x) if x else "",
                    return_dtype=pl.Utf8,
                ).alias("Tags")
            )

        return focus_data.write_csv()

    async def export_to_dict(
        self,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Export usage data to FOCUS format as a dictionary.

        Args:
            limit: Optional limit on number of records
            start_time_utc: Optional start time filter
            end_time_utc: Optional end time filter

        Returns:
            Dictionary with FOCUS formatted data and summary statistics
        """
        focus_data = await self.get_focus_data(
            limit=limit,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
        )

        if focus_data.is_empty():
            return {
                "focus_version": "1.0",
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "records": [],
                "summary": {
                    "total_records": 0,
                    "total_billed_cost": 0.0,
                    "total_consumed_quantity": 0,
                    "unique_providers": 0,
                    "unique_sub_accounts": 0,
                },
            }

        records = focus_data.to_dicts()

        # Calculate summary statistics
        total_cost = sum(r.get("BilledCost", 0) or 0 for r in records)
        total_tokens = sum(r.get("ConsumedQuantity", 0) or 0 for r in records)
        unique_providers = len(
            set(r.get("ProviderName", "") for r in records if r.get("ProviderName"))
        )
        unique_sub_accounts = len(
            set(r.get("SubAccountId", "") for r in records if r.get("SubAccountId"))
        )

        return {
            "focus_version": "1.0",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "records": records,
            "summary": {
                "total_records": len(records),
                "total_billed_cost": total_cost,
                "total_consumed_quantity": total_tokens,
                "unique_providers": unique_providers,
                "unique_sub_accounts": unique_sub_accounts,
            },
        }

    async def dry_run_export(
        self,
        limit: Optional[int] = 10000,
    ) -> Dict[str, Any]:
        """
        Perform a dry run export, returning data without writing to any destination.

        This is useful for testing and previewing the FOCUS output.

        Args:
            limit: Maximum number of records to include (default: 10000)

        Returns:
            Dictionary containing raw data sample, FOCUS data, and summary
        """
        from litellm.integrations.focus.database import FOCUSDatabase
        from litellm.integrations.focus.transform import FOCUSTransformer

        try:
            verbose_logger.debug("FOCUS Exporter: Starting dry run export")

            # Load data from database
            database = FOCUSDatabase()
            data = await database.get_usage_data(limit=limit)

            if data.is_empty():
                verbose_logger.debug("FOCUS Dry Run: No usage data found")
                return {
                    "raw_data_sample": [],
                    "focus_data": [],
                    "summary": {
                        "total_records": 0,
                        "total_billed_cost": 0.0,
                        "total_consumed_quantity": 0,
                        "unique_providers": 0,
                        "unique_sub_accounts": 0,
                    },
                }

            verbose_logger.debug(f"FOCUS Dry Run: Processing {len(data)} records")

            # Sample of raw data
            raw_data_sample = data.head(50).to_dicts()

            # Transform to FOCUS format
            transformer = FOCUSTransformer(
                include_tags=self.include_tags,
                include_token_breakdown=self.include_token_breakdown,
            )
            focus_data = transformer.transform(data)

            if focus_data.is_empty():
                verbose_logger.debug(
                    "FOCUS Dry Run: No valid data after transformation"
                )
                return {
                    "raw_data_sample": raw_data_sample,
                    "focus_data": [],
                    "summary": {
                        "total_records": 0,
                        "total_billed_cost": sum(
                            r.get("spend", 0) or 0 for r in raw_data_sample
                        ),
                        "total_consumed_quantity": sum(
                            (r.get("prompt_tokens", 0) or 0)
                            + (r.get("completion_tokens", 0) or 0)
                            for r in raw_data_sample
                        ),
                        "unique_providers": 0,
                        "unique_sub_accounts": 0,
                    },
                }

            focus_records = focus_data.to_dicts()

            # Calculate summary
            total_cost = sum(r.get("BilledCost", 0) or 0 for r in focus_records)
            total_tokens = sum(
                r.get("ConsumedQuantity", 0) or 0 for r in focus_records
            )
            unique_providers = len(
                set(
                    r.get("ProviderName", "")
                    for r in focus_records
                    if r.get("ProviderName")
                )
            )
            unique_sub_accounts = len(
                set(
                    r.get("SubAccountId", "")
                    for r in focus_records
                    if r.get("SubAccountId")
                )
            )

            verbose_logger.debug(
                f"FOCUS Dry Run: Completed with {len(focus_records)} records"
            )

            return {
                "raw_data_sample": raw_data_sample,
                "focus_data": focus_records,
                "summary": {
                    "total_records": len(focus_records),
                    "total_billed_cost": total_cost,
                    "total_consumed_quantity": total_tokens,
                    "unique_providers": unique_providers,
                    "unique_sub_accounts": unique_sub_accounts,
                },
            }

        except Exception as e:
            verbose_logger.error(f"FOCUS Dry Run: Error: {e}")
            raise
