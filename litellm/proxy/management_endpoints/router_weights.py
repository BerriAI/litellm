from abc import abstractmethod
from collections.abc import Mapping
from typing import Annotated, Final, Protocol

from fastapi import HTTPException
from pydantic import BaseModel, BeforeValidator, ValidationError

from litellm.repositories.prisma_protocols import TableActions
from litellm.types.router_weights import RouterWeights


class _StoredModel(Protocol):
    @property
    @abstractmethod
    def model_id(self) -> str:
        pass


class _ModelDb(Protocol):
    @property
    @abstractmethod
    def litellm_proxymodeltable(self) -> TableActions[_StoredModel]:
        pass


class _PrismaClient(Protocol):
    @property
    @abstractmethod
    def replica_db(self) -> _ModelDb:
        pass


class _Router(Protocol):
    @abstractmethod
    def get_deployment(self, model_id: str) -> object | None:
        pass


class _RouterWeightSettings(BaseModel):
    weights: RouterWeights | None = None


class _RouterWeightModelInfo(BaseModel):
    team_id: str | None = None
    db_model: bool | None = None
    team_public_model_name: str | None = None


def _router_weight_model_info(value: object) -> _RouterWeightModelInfo:
    if isinstance(value, str):
        return _RouterWeightModelInfo.model_validate_json(value)
    return _RouterWeightModelInfo.model_validate(value or {}, from_attributes=True)


class _RouterWeightDeployment(BaseModel):
    model_name: str
    model_info: Annotated[_RouterWeightModelInfo, BeforeValidator(_router_weight_model_info)]


def _validate_router_weight_reference(
    model_group: str,
    deployment_id: str,
    team_id: str | None,
    stored: _RouterWeightDeployment | None,
    configured: object | None,
) -> None:
    reference: Final = (
        stored
        if stored is not None
        else (
            _RouterWeightDeployment.model_validate(configured, from_attributes=True) if configured is not None else None
        )
    )
    if (
        reference is None
        or (stored is None and reference.model_info.db_model)
        or (reference.model_info.team_id is not None and reference.model_info.team_id != team_id)
    ):
        raise HTTPException(status_code=400, detail=f"Unknown deployment ID in router weights: {deployment_id}")
    canonical_group: Final = (
        reference.model_info.team_public_model_name if reference.model_info.team_id is not None else None
    ) or reference.model_name
    if model_group != canonical_group:
        raise HTTPException(
            status_code=400,
            detail=f"Deployment {deployment_id} does not belong to model group {model_group}",
        )


async def validate_router_settings_weights(
    router_settings: BaseModel | Mapping[str, object] | None,
    *,
    team_id: str | None,
    prisma_client: _PrismaClient | None,
    llm_router: _Router | None,
) -> None:
    try:
        weights: Final = (
            _RouterWeightSettings.model_validate(router_settings, from_attributes=True).weights
            if router_settings is not None
            else None
        )
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail="Invalid router weights. Replace or clear router_settings.weights.",
        ) from None
    if not weights:
        return
    deployment_ids: Final = frozenset(deployment_id for group in weights.values() for deployment_id in group)
    if not deployment_ids:
        return
    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable while validating router weights")
    stored_models: Final = await prisma_client.replica_db.litellm_proxymodeltable.find_many(
        where={"model_id": {"in": list(deployment_ids)}}
    )
    stored_by_id: Final = {
        row.model_id: _RouterWeightDeployment.model_validate(row, from_attributes=True) for row in stored_models
    }
    for model_group, group_weights in weights.items():
        for deployment_id in group_weights:
            _validate_router_weight_reference(
                model_group,
                deployment_id,
                team_id,
                stored_by_id.get(deployment_id),
                llm_router.get_deployment(model_id=deployment_id) if llm_router is not None else None,
            )
