"""
Subscription accounts: the operator-entered recurring fee of a provider account whose
requests are covered by a flat subscription, shown beside metered spend on the Usage page.
"""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Query, status

from litellm.constants import LITELLM_PROXY_ADMIN_NAME, SUBSCRIPTION_BACKED_PROVIDERS
from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.models.subscription_account import LiteLLM_SubscriptionAccountTable
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.spend_tracking.subscription_billing_periods import billing_periods_starting_within
from litellm.proxy.utils import get_prisma_client_or_throw
from litellm.repositories.subscription_account_repository import (
    DuplicateSubscriptionAccount,
    KeepLabel,
    SetLabel,
    SubscriptionAccountFeeChanges,
    SubscriptionAccountFeeFields,
    SubscriptionAccountRepository,
    SubscriptionAccountStore,
)
from litellm.types.proxy.management_endpoints.subscription_accounts import (
    BillingPeriod,
    DeleteSubscriptionAccountRequest,
    SubscriptionAccountFeeRequest,
    SubscriptionAccountUsage,
    SubscriptionUsageResponse,
    UpdateSubscriptionAccountFeeRequest,
)
from litellm.types.router import DeploymentTypedDict

router: Final = APIRouter(prefix="/subscription_accounts", tags=["subscription accounts"])
Auth: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]


@dataclass(frozen=True, slots=True)
class SubscriptionDeployment:
    model_name: str
    custom_llm_provider: str
    account_id: str | None


AccountIdResolver = Callable[[str], str | None]


def resolve_subscription_account_id(custom_llm_provider: str) -> str | None:
    return Authenticator().get_account_id() if custom_llm_provider == "chatgpt" else None


def _declared_provider(litellm_params: Mapping[str, object]) -> str | None:
    model: Final = litellm_params.get("model")
    configured_provider: Final = litellm_params.get("custom_llm_provider")
    return declared_authenticating_provider(
        model if isinstance(model, str) else None,
        configured_provider if isinstance(configured_provider, str) else None,
    )


def _subscription_backed_entry(entry: DeploymentTypedDict) -> tuple[str, str] | None:
    provider: Final = _declared_provider(entry["litellm_params"])
    if provider is None or provider not in SUBSCRIPTION_BACKED_PROVIDERS:
        return None
    return entry["model_name"], provider


def subscription_deployments_from_model_list(
    model_list: Iterable[DeploymentTypedDict], account_id_resolver: AccountIdResolver
) -> tuple[SubscriptionDeployment, ...]:
    candidates: Final = (_subscription_backed_entry(entry) for entry in model_list)
    entries: Final = tuple(candidate for candidate in candidates if candidate is not None)
    account_ids: Final = {provider: account_id_resolver(provider) for provider in {p for _, p in entries}}
    return tuple(
        SubscriptionDeployment(model_name=model_name, custom_llm_provider=provider, account_id=account_ids[provider])
        for model_name, provider in entries
    )


def get_subscription_deployments() -> tuple[SubscriptionDeployment, ...]:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return ()
    return subscription_deployments_from_model_list(llm_router.get_model_list() or [], resolve_subscription_account_id)


def get_subscription_account_repository() -> SubscriptionAccountStore:
    return SubscriptionAccountRepository(get_prisma_client_or_throw(CommonProxyErrors.db_not_connected_error.value))


Store: TypeAlias = Annotated[SubscriptionAccountStore, Depends(get_subscription_account_repository)]
Deployments: TypeAlias = Annotated[tuple[SubscriptionDeployment, ...], Depends(get_subscription_deployments)]


def _forbidden(user_api_key_dict: UserAPIKeyAuth) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
    )


def _account_usage(
    custom_llm_provider: str,
    account_id: str | None,
    deployments: Iterable[SubscriptionDeployment],
    fee_row: LiteLLM_SubscriptionAccountTable | None,
    start_date: date,
    end_date: date,
) -> SubscriptionAccountUsage:
    model_names: Final = sorted({deployment.model_name for deployment in deployments})
    if fee_row is None:
        return SubscriptionAccountUsage(
            custom_llm_provider=custom_llm_provider,
            account_id=account_id,
            deployments=model_names,
            fee=None,
            billing_periods=[],
            fixed_cost=None,
        )
    periods: Final = billing_periods_starting_within(
        date.fromisoformat(fee_row.billing_period_start), start_date, end_date
    )
    return SubscriptionAccountUsage(
        custom_llm_provider=custom_llm_provider,
        account_id=account_id,
        deployments=model_names,
        fee=fee_row,
        billing_periods=[BillingPeriod(start=period.start, end=period.end) for period in periods],
        fixed_cost=len(periods) * fee_row.monthly_fee,
    )


def build_subscription_usage(
    deployments: Iterable[SubscriptionDeployment],
    fee_rows: Iterable[LiteLLM_SubscriptionAccountTable],
    start_date: date,
    end_date: date,
) -> SubscriptionUsageResponse:
    deployment_tuple: Final = tuple(deployments)
    fees_by_account: Final = {(row.custom_llm_provider, row.account_id): row for row in fee_rows}
    deployment_keys: Final = {(d.custom_llm_provider, d.account_id) for d in deployment_tuple}
    account_keys: Final = sorted(deployment_keys | set(fees_by_account), key=lambda key: (key[0], key[1] or "~"))
    accounts: Final = [
        _account_usage(
            provider,
            account_id,
            (d for d in deployment_tuple if (d.custom_llm_provider, d.account_id) == (provider, account_id)),
            fees_by_account.get((provider, account_id)) if account_id is not None else None,
            start_date,
            end_date,
        )
        for provider, account_id in account_keys
    ]
    return SubscriptionUsageResponse(
        start_date=start_date,
        end_date=end_date,
        subscription_providers=sorted(SUBSCRIPTION_BACKED_PROVIDERS),
        accounts=accounts,
    )


_DAY_PATTERN: Final = re.compile(r"\d{4}-\d{2}-\d{2}")


def _parse_day(value: str) -> date | None:
    if _DAY_PATTERN.fullmatch(value) is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_date(value: str, name: str) -> date:
    parsed: Final = _parse_day(value)
    if parsed is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail={"error": f"{name} must be YYYY-MM-DD, got {value!r}"}
        )
    return parsed


@router.get("/usage", response_model=SubscriptionUsageResponse, dependencies=[Depends(user_api_key_auth)])
async def get_subscription_usage(
    start_date: Annotated[str, Query(description="First day of the Usage page filter, YYYY-MM-DD")],
    end_date: Annotated[str, Query(description="Last day of the Usage page filter, YYYY-MM-DD")],
    user_api_key_dict: Auth,
    repository: Store,
    deployments: Deployments,
) -> SubscriptionUsageResponse:
    """
    Subscription-backed accounts behind the proxy's deployments, each with its configured
    monthly fee and the fixed cost attributed to the date range: the fee once per billing
    period whose start date falls inside the range, however many deployments share the account.
    """
    if not user_api_key_has_admin_view(user_api_key_dict):
        raise _forbidden(user_api_key_dict)
    start: Final = _parse_date(start_date, "start_date")
    end: Final = _parse_date(end_date, "end_date")
    if end < start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "end_date must not be before start_date"}
        )
    return build_subscription_usage(deployments, await repository.list_all(), start, end)


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise _forbidden(user_api_key_dict)


def _actor(user_api_key_dict: UserAPIKeyAuth) -> str:
    return user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME


@router.post("/new", response_model=LiteLLM_SubscriptionAccountTable, dependencies=[Depends(user_api_key_auth)])
async def new_subscription_account(
    request: SubscriptionAccountFeeRequest,
    user_api_key_dict: Auth,
    repository: Store,
) -> LiteLLM_SubscriptionAccountTable:
    """Record the monthly fee of one subscription-backed provider account."""
    _require_proxy_admin(user_api_key_dict)
    if request.custom_llm_provider not in SUBSCRIPTION_BACKED_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"{request.custom_llm_provider!r} is not a subscription-backed provider; "
                f"expected one of {sorted(SUBSCRIPTION_BACKED_PROVIDERS)}"
            },
        )
    created: Final = await repository.create_account(
        request.custom_llm_provider,
        request.account_id,
        SubscriptionAccountFeeFields(
            monthly_fee=request.monthly_fee,
            currency=request.currency,
            billing_period_start=request.billing_period_start.isoformat(),
            label=request.label,
        ),
        _actor(user_api_key_dict),
    )
    if isinstance(created, DuplicateSubscriptionAccount):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"A fee is already configured for {created.custom_llm_provider} account "
                f"{created.account_id}; use /subscription_accounts/update to change it"
            },
        )
    return created


def _fee_changes(request: UpdateSubscriptionAccountFeeRequest) -> SubscriptionAccountFeeChanges:
    return SubscriptionAccountFeeChanges(
        monthly_fee=request.monthly_fee,
        currency=request.currency,
        billing_period_start=(
            request.billing_period_start.isoformat() if request.billing_period_start is not None else None
        ),
        label=SetLabel(request.label) if "label" in request.model_fields_set else KeepLabel(),
    )


@router.post("/update", response_model=LiteLLM_SubscriptionAccountTable, dependencies=[Depends(user_api_key_auth)])
async def update_subscription_account(
    request: UpdateSubscriptionAccountFeeRequest,
    user_api_key_dict: Auth,
    repository: Store,
) -> LiteLLM_SubscriptionAccountTable:
    """
    Change the fee, currency, billing period start, or label of a subscription account.
    Only the fields named in the request are written, so two admins changing different
    fields at the same time both land.
    """
    _require_proxy_admin(user_api_key_dict)
    updated: Final = await repository.update_fee(
        request.subscription_account_id, _fee_changes(request), _actor(user_api_key_dict)
    )
    if updated is None:
        raise _not_found(request.subscription_account_id)
    return updated


@router.post("/delete", response_model=LiteLLM_SubscriptionAccountTable, dependencies=[Depends(user_api_key_auth)])
async def delete_subscription_account(
    request: DeleteSubscriptionAccountRequest,
    user_api_key_dict: Auth,
    repository: Store,
) -> LiteLLM_SubscriptionAccountTable:
    """Stop tracking a subscription account's fee; its requests stay marked as subscription-covered."""
    _require_proxy_admin(user_api_key_dict)
    if await repository.find_by_id(request.subscription_account_id) is None:
        raise _not_found(request.subscription_account_id)
    deleted: Final = await repository.delete_account(request.subscription_account_id)
    if deleted is None:
        raise _not_found(request.subscription_account_id)
    return deleted


def _not_found(subscription_account_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": f"No subscription account with id {subscription_account_id}"},
    )
