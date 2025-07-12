from pydantic import Field

from ...router import ModelGroupInfo


class ModelGroupInfoProxy(ModelGroupInfo):
    is_public_model_group: bool = Field(default=False)
