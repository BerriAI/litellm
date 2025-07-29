from typing import Dict

from pydantic import BaseModel, Field

from ...router import ModelGroupInfo


class ModelGroupInfoProxy(ModelGroupInfo):
    is_public_model_group: bool = Field(default=False)


class UpdateUsefulLinksRequest(BaseModel):
    useful_links: Dict[str, str]
