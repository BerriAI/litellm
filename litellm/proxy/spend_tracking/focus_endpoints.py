"""
FOCUS (FinOps Open Cost & Usage Specification) endpoints for LiteLLM Proxy.

Provides endpoints to export LiteLLM usage data in FOCUS format for
interoperability with FinOps tools like APTIO.

More info: https://focus.finops.org/
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.focus_endpoints import (
    FOCUSDryRunResponse,
    FOCUSExportRequest,
    FOCUSExportResponse,
    FOCUSSummary,
)

router = APIRouter()


@router.post(
    "/focus/export",
    tags=["FOCUS"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=FOCUSExportResponse,
)
async def focus_export(
    request: FOCUSExportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Export usage data in FOCUS (FinOps Open Cost & Usage Specification) format.

    FOCUS is an open specification for consistent cost and usage datasets from
    the FinOps Foundation. This endpoint exports LiteLLM usage data in a format
    compatible with FinOps tools like APTIO.

    Parameters:
    - limit: Optional limit on number of records to export
    - start_time_utc: Optional start time filter in UTC
    - end_time_utc: Optional end time filter in UTC
    - include_tags: Whether to include tags (default: true)
    - include_token_breakdown: Whether to include token breakdown in tags (default: true)

    Returns:
    - FOCUS formatted data as JSON with summary statistics

    Only admin users can perform FOCUS exports.
    """
    # Validation - only admins can export cost data
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        from litellm.integrations.focus.focus import FOCUSExporter

        # Initialize exporter with configuration
        exporter = FOCUSExporter(
            include_tags=request.include_tags,
            include_token_breakdown=request.include_token_breakdown,
        )

        # Export data
        result = await exporter.export_to_dict(
            limit=request.limit,
            start_time_utc=request.start_time_utc,
            end_time_utc=request.end_time_utc,
        )

        verbose_proxy_logger.info(
            f"FOCUS export completed: {result['summary']['total_records']} records"
        )

        return FOCUSExportResponse(
            message="FOCUS export completed successfully",
            status="success",
            format="json",
            data=result,
            summary=FOCUSSummary(**result["summary"]),
        )

    except Exception as e:
        verbose_proxy_logger.error(f"Error performing FOCUS export: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to perform FOCUS export: {str(e)}"},
        )


@router.post(
    "/focus/export/csv",
    tags=["FOCUS"],
    dependencies=[Depends(user_api_key_auth)],
    response_class=PlainTextResponse,
)
async def focus_export_csv(
    request: FOCUSExportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Export usage data in FOCUS format as CSV.

    This endpoint exports LiteLLM usage data in FOCUS CSV format,
    suitable for importing into spreadsheets or FinOps tools.

    Parameters:
    - limit: Optional limit on number of records to export
    - start_time_utc: Optional start time filter in UTC
    - end_time_utc: Optional end time filter in UTC
    - include_tags: Whether to include tags (default: true)
    - include_token_breakdown: Whether to include token breakdown in tags (default: true)

    Returns:
    - FOCUS formatted data as CSV

    Only admin users can perform FOCUS exports.
    """
    # Validation - only admins can export cost data
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        from litellm.integrations.focus.focus import FOCUSExporter

        # Initialize exporter
        exporter = FOCUSExporter(
            include_tags=request.include_tags,
            include_token_breakdown=request.include_token_breakdown,
        )

        # Export as CSV
        csv_data = await exporter.export_csv(
            limit=request.limit,
            start_time_utc=request.start_time_utc,
            end_time_utc=request.end_time_utc,
        )

        if not csv_data:
            return PlainTextResponse(
                content="No data to export",
                status_code=200,
                media_type="text/csv",
            )

        verbose_proxy_logger.info("FOCUS CSV export completed")

        return PlainTextResponse(
            content=csv_data,
            status_code=200,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=focus_export.csv"
            },
        )

    except Exception as e:
        verbose_proxy_logger.error(f"Error performing FOCUS CSV export: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to perform FOCUS CSV export: {str(e)}"},
        )


@router.post(
    "/focus/dry-run",
    tags=["FOCUS"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=FOCUSDryRunResponse,
)
async def focus_dry_run(
    request: FOCUSExportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Perform a dry run export in FOCUS format.

    This endpoint performs a dry run export, returning the data that would be
    exported without any side effects. Useful for testing and previewing.

    Parameters:
    - limit: Optional limit on number of records (default: 10000)
    - include_tags: Whether to include tags (default: true)
    - include_token_breakdown: Whether to include token breakdown in tags (default: true)

    Returns:
    - raw_data_sample: Sample of raw database records (first 50)
    - focus_data: FOCUS formatted data
    - summary: Statistics including total cost, tokens, and record counts

    Only admin users can perform FOCUS exports.
    """
    # Validation - only admins can export cost data
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        from litellm.integrations.focus.focus import FOCUSExporter

        # Initialize exporter
        exporter = FOCUSExporter(
            include_tags=request.include_tags,
            include_token_breakdown=request.include_token_breakdown,
        )

        # Perform dry run
        result = await exporter.dry_run_export(
            limit=request.limit or 10000,
        )

        verbose_proxy_logger.info("FOCUS dry run export completed")

        return FOCUSDryRunResponse(
            message="FOCUS dry run export completed successfully",
            status="success",
            raw_data_sample=result.get("raw_data_sample"),
            focus_data=result.get("focus_data"),
            summary=FOCUSSummary(**result["summary"]) if result.get("summary") else None,
        )

    except Exception as e:
        verbose_proxy_logger.error(f"Error performing FOCUS dry run: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to perform FOCUS dry run: {str(e)}"},
        )


@router.get(
    "/focus/schema",
    tags=["FOCUS"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_focus_schema(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get the FOCUS schema documentation.

    Returns information about the FOCUS columns exported by LiteLLM.

    Only admin users can access this endpoint.
    """
    # Validation - only admins can access
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    return {
        "focus_version": "1.0",
        "specification_url": "https://focus.finops.org/",
        "columns": {
            "required": {
                "BilledCost": "The cost that is invoiced/billed (float)",
                "BillingPeriodStart": "Start of billing period (ISO datetime)",
                "BillingPeriodEnd": "End of billing period (ISO datetime)",
            },
            "recommended": {
                "ChargeCategory": "Type of charge (Usage)",
                "ChargeClass": "Classification of charge (Standard)",
                "ChargeDescription": "Human-readable description",
                "ChargePeriodStart": "Start of charge period (ISO datetime)",
                "ChargePeriodEnd": "End of charge period (ISO datetime)",
                "ConsumedQuantity": "Amount consumed (tokens)",
                "ConsumedUnit": "Unit of consumption (Tokens)",
                "EffectiveCost": "Cost after discounts",
                "ListCost": "Cost at list prices",
                "ProviderName": "LLM provider name (OpenAI, Anthropic, etc.)",
                "PublisherName": "Publisher name (LiteLLM)",
                "Region": "Geographic region (if applicable)",
                "ResourceId": "Unique resource identifier",
                "ResourceName": "Model name",
                "ResourceType": "Type of resource (LLM)",
                "ServiceCategory": "Service category (AI and Machine Learning)",
                "ServiceName": "Service name (LLM Inference)",
                "SubAccountId": "Sub-account ID (team_id)",
                "SubAccountName": "Sub-account name (team_alias)",
                "Tags": "Additional metadata tags",
            },
        },
        "litellm_tags": {
            "litellm:provider": "LLM provider identifier",
            "litellm:model": "Model identifier",
            "litellm:model_group": "Model group name",
            "litellm:user_id": "User ID",
            "litellm:api_key_prefix": "First 8 chars of API key",
            "litellm:api_key_alias": "API key alias",
            "litellm:prompt_tokens": "Number of prompt tokens",
            "litellm:completion_tokens": "Number of completion tokens",
            "litellm:api_requests": "Number of API requests",
            "litellm:successful_requests": "Number of successful requests",
            "litellm:failed_requests": "Number of failed requests",
            "litellm:cache_creation_tokens": "Cache creation tokens",
            "litellm:cache_read_tokens": "Cache read tokens",
        },
    }
