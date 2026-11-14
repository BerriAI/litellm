from typing import Annotated, TypeAlias

from fastapi import Depends

from litellm.models.alert import AlertStore
from litellm.models.deployment_warning import DeploymentWarningIgnoreStore
from litellm.models.fallback import RouterFallbackStore
from litellm.models.gateway_activity import GatewayActivityStore
from litellm.models.performance_alert import AlertMonitoringStore
from litellm.models.plugin_upload import PluginUploadStore
from litellm.models.skill_submission import SkillSubmissionStore
from litellm.models.user_banner import UserBannerStore
from litellm.proxy.deps.database import DatabaseDep
from litellm.repositories.alert_repository import AlertRepository
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.deployment_warning_ignore_repository import DeploymentWarningIgnoreRepository
from litellm.repositories.gateway_activity_repository import GatewayActivityRepository
from litellm.repositories.performance_alert_repository import PerformanceAlertRepository
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


async def get_alert_repository(database: DatabaseDep) -> AlertStore | None:
    return None if database is None else AlertRepository(database.prisma)


AlertRepositoryDep: TypeAlias = Annotated[AlertStore | None, Depends(get_alert_repository)]


async def get_alert_monitoring_repository(database: DatabaseDep) -> AlertMonitoringStore | None:
    from litellm.proxy.proxy_server import general_settings_view, premium_user

    return (
        None
        if database is None
        else PerformanceAlertRepository(
            database.prisma,
            enterprise=premium_user is True,
            configured=general_settings_view().get("performance_alerts_enabled", True) is True,
        )
    )


AlertMonitoringRepositoryDep: TypeAlias = Annotated[
    AlertMonitoringStore | None, Depends(get_alert_monitoring_repository)
]


async def get_router_fallback_repository(database: DatabaseDep) -> RouterFallbackStore | None:
    return None if database is None else ConfigRepository(database.prisma, use_writer=True)


RouterFallbackRepositoryDep: TypeAlias = Annotated[RouterFallbackStore | None, Depends(get_router_fallback_repository)]
