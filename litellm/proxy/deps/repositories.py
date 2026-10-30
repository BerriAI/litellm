from typing import Annotated, TypeAlias

from fastapi import Depends

from litellm.models.user_banner import UserBannerStore
from litellm.proxy.deps.database import DatabaseDep
from litellm.repositories.user_banner_repository import UserBannerRepository


async def get_user_banner_repository(database: DatabaseDep) -> UserBannerStore | None:
    return None if database is None else UserBannerRepository(database.prisma)


UserBannerRepositoryDep: TypeAlias = Annotated[UserBannerStore | None, Depends(get_user_banner_repository)]
