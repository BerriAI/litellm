from typing import Protocol, assert_never

from litellm._auth.decision import (
    Allow,
    Deny,
    EvictPresentedKey,
    LoadKey,
    Outcome,
    PresentedKey,
    ResolveEndUser,
    ResolvePresentedKey,
    Step,
)


class EffectHandler(Protocol):
    async def resolve_end_user(self, prefetch_key: bool) -> None: ...

    async def resolve_presented_key(self, hashed_token: str) -> PresentedKey | None: ...

    async def evict_presented_key(self, hashed_token: str) -> None: ...

    async def load_key(self, hashed_token: str) -> None: ...


async def drive(step: Step, handler: EffectHandler) -> Outcome:
    match step:
        case Allow() | Deny():
            return step
        case ResolveEndUser(prefetch_key=prefetch_key, resume=resume):
            await handler.resolve_end_user(prefetch_key)
            return await drive(resume(), handler)
        case ResolvePresentedKey(hashed_token=hashed_token, resume=resume):
            return await drive(resume(await handler.resolve_presented_key(hashed_token)), handler)
        case EvictPresentedKey(hashed_token=hashed_token, resume=resume):
            await handler.evict_presented_key(hashed_token)
            return await drive(resume(), handler)
        case LoadKey(hashed_token=hashed_token, resume=resume):
            await handler.load_key(hashed_token)
            return await drive(resume(), handler)
        case _:
            assert_never(step)
