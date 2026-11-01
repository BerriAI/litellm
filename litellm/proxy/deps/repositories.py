from typing import Annotated, TypeAlias

from fastapi import Depends

from litellm.models.deployment_warning import DeploymentWarningIgnoreStore
from litellm.models.gateway_activity import GatewayActivityStore
from litellm.models.user_banner import UserBannerStore
from litellm.models.verification_token import KeyRevocationStore
from litellm.proxy.deps.database import DatabaseDep
from litellm.repositories.deployment_warning_ignore_repository import DeploymentWarningIgnoreRepository
from litellm.repositories.gateway_activity_repository import GatewayActivityRepository
from litellm.repositories.user_banner_repository import UserBannerRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository


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


async def get_verification_token_repository(database: DatabaseDep) -> KeyRevocationStore | None:
    return None if database is None else VerificationTokenRepository(database.prisma, use_writer=True)


VerificationTokenRepositoryDep: TypeAlias = Annotated[
    KeyRevocationStore | None, Depends(get_verification_token_repository)
]
