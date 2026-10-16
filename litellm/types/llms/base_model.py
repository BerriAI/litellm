import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from litellm.constants import DEFER_PYDANTIC_BUILD

_SCHEMA_BUILD_LOCK: Final = threading.RLock()


class LiteLLMBaseModel(BaseModel):
    model_config = ConfigDict(defer_build=DEFER_PYDANTIC_BUILD)

    if TYPE_CHECKING:

        def __init__(self, /, **data: object) -> None: ...

    @classmethod
    def model_rebuild(
        cls,
        *,
        force: bool = False,
        raise_errors: bool = True,
        _parent_namespace_depth: int = 2,
        _types_namespace: Mapping[str, object] | None = None,
    ) -> bool | None:
        # Resolve names from the model's own module, never a caller frame: a deferred first-use build
        # reads f_locals 5 frames up, and on Python < 3.13 that rewrites the dict the caller's locals() returned
        with _SCHEMA_BUILD_LOCK:
            return super().model_rebuild(
                force=force,
                raise_errors=raise_errors,
                _parent_namespace_depth=0,
                _types_namespace=_types_namespace,
            )

    def model_post_init(self, context: object, /) -> None:
        # Instances built by a parent's validator or by model_construct skip this class's own
        # first-use build, leaving a MockValSer that fails when the instance is serialized as Any
        # TODO: drop once pydantic ships a fix for https://github.com/pydantic/pydantic/issues/13647
        model = type(self)
        if not model.__pydantic_complete__:
            model.model_rebuild(raise_errors=False)
