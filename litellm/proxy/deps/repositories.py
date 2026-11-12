from typing import Annotated, TypeAlias

from fastapi import Depends

from litellm.models.deployment_warning import DeploymentWarningIgnoreStore
from litellm.models.fallback import RouterFallbackStore
from litellm.models.gateway_activity import GatewayActivityStore
from litellm.models.plugin_upload import PluginUploadStore
from litellm.models.skill_submission import SkillSubmissionStore
from litellm.models.user_banner import UserBannerStore
from litellm.proxy.deps.database import DatabaseDep
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.deployment_warning_ignore_repository import DeploymentWarningIgnoreRepository
from litellm.repositories.gateway_activity_repository import GatewayActivityRepository
from litellm.repositories.plugin_upload_repository import PluginUploadRepository
from litellm.repositories.skill_submission_repository import SkillSubmissionRepository
from litellm.repositories.user_banner_repository import UserBannerRepository


async def get_user_banner_repository(database: DatabaseDep) -> UserBannerStore | None:
    return None if database is None else UserBannerRepository(database.prisma)


UserBannerRepositoryDep: TypeAlias = Annotated[UserBannerStore | None, Depends(get_user_banner_repository)]


async def get_deployment_warning_ignore_repository(database: DatabaseDep) -> DeploymentWarningIgnoreStore | None:
    return None if database is None else DeploymentWarningIgnoreRepository(database.prisma, use_writer=True)


DeploymentWarningIgnoreRepositoryDep: TypeAlias = Annotated[
    DeploymentWarningIgnoreStore | None, Depends(get_deployment_warning_ignore_repository)
]


async def get_gateway_activity_repository(database: DatabaseDep) -> GatewayActivityStore | None:
    return None if database is None else GatewayActivityRepository(database.prisma)


GatewayActivityRepositoryDep: TypeAlias = Annotated[
    GatewayActivityStore | None, Depends(get_gateway_activity_repository)
]


async def get_plugin_upload_repository(database: DatabaseDep) -> PluginUploadStore | None:
    return None if database is None else PluginUploadRepository(database.prisma)


PluginUploadRepositoryDep: TypeAlias = Annotated[PluginUploadStore | None, Depends(get_plugin_upload_repository)]


async def get_skill_submission_repository(database: DatabaseDep) -> SkillSubmissionStore | None:
    return None if database is None else SkillSubmissionRepository(database.prisma)


SkillSubmissionRepositoryDep: TypeAlias = Annotated[
    SkillSubmissionStore | None, Depends(get_skill_submission_repository)
]


async def get_router_fallback_repository(database: DatabaseDep) -> RouterFallbackStore | None:
    return None if database is None else ConfigRepository(database.prisma, use_writer=True)


RouterFallbackRepositoryDep: TypeAlias = Annotated[RouterFallbackStore | None, Depends(get_router_fallback_repository)]
