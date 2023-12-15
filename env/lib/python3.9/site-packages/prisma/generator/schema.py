from enum import Enum
from typing import Any, Dict, List, Tuple, Type, Union
from typing_extensions import ClassVar

from pydantic import BaseModel

from .models import AnyData, Model as ModelInfo, PrimaryKey
from .._compat import (
    PYDANTIC_V2,
    ConfigDict,
    root_validator,
    cached_property,
    model_rebuild,
)


class Kind(str, Enum):
    alias = 'alias'
    union = 'union'
    typeddict = 'typeddict'


class PrismaType(BaseModel):
    name: str
    kind: Kind
    subtypes: List['PrismaType'] = []

    @classmethod
    def from_subtypes(
        cls, subtypes: List['PrismaType'], **kwargs: Any
    ) -> Union['PrismaUnion', 'PrismaAlias']:
        """Return either a `PrismaUnion` or a `PrismaAlias` depending on the number of subtypes"""
        if len(subtypes) > 1:
            return PrismaUnion(subtypes=subtypes, **kwargs)

        return PrismaAlias(subtypes=subtypes, **kwargs)


class PrismaDict(PrismaType):
    kind: Kind = Kind.typeddict
    fields: Dict[str, str]
    total: bool


class PrismaUnion(PrismaType):
    kind: Kind = Kind.union
    subtypes: List[PrismaType]


class PrismaAlias(PrismaType):
    kind: Kind = Kind.alias
    to: str

    @root_validator(pre=True)
    @classmethod
    def transform_to(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if 'to' not in values and 'subtypes' in values:
            values['to'] = values['subtypes'][0].name
        return values


class Schema(BaseModel):
    models: List['Model']

    @classmethod
    def from_data(cls, data: AnyData) -> 'Schema':
        models = [Model(info=model) for model in data.dmmf.datamodel.models]
        return cls(models=models)

    def get_model(self, name: str) -> 'Model':
        for model in self.models:
            if model.info.name == name:
                return model

        raise LookupError(f'Unknown model: {name}')


class Model(BaseModel):
    info: ModelInfo

    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(
            ignored_types=(cached_property,)
        )
    else:

        class Config:
            keep_untouched: Tuple[Type[Any], ...] = (cached_property,)

    @cached_property
    def where_unique(self) -> PrismaType:
        info = self.info
        model = info.name
        subtypes: List[PrismaType] = [
            PrismaDict(
                total=True,
                name=f'_{model}WhereUnique_{field.name}_Input',
                fields={
                    field.name: field.python_type,
                },
            )
            for field in info.scalar_fields
            if field.is_id or field.is_unique
        ]

        for key in [info.compound_primary_key, *info.unique_indexes]:
            if key is None:
                continue

            if isinstance(key, PrimaryKey):
                name = f'_{model}CompoundPrimaryKey'
            else:
                name = f'_{model}Compound{key.name}Key'

            subtypes.append(
                PrismaDict(
                    name=name,
                    total=True,
                    fields={
                        key.name: f'{name}Inner',
                    },
                    subtypes=[
                        PrismaDict(
                            total=True,
                            name=f'{name}Inner',
                            fields={
                                field.name: field.python_type
                                for field in map(
                                    info.resolve_field, key.fields
                                )
                            },
                        )
                    ],
                )
            )

        return PrismaType.from_subtypes(
            subtypes, name=f'{model}WhereUniqueInput'
        )

    @cached_property
    def order_by(self) -> PrismaType:
        model = self.info.name
        subtypes: List[PrismaType] = [
            PrismaDict(
                name=f'_{model}_{field.name}_OrderByInput',
                total=True,
                fields={
                    field.name: 'SortOrder',
                },
            )
            for field in self.info.scalar_fields
        ]
        return PrismaType.from_subtypes(subtypes, name=f'{model}OrderByInput')


model_rebuild(Schema)
model_rebuild(PrismaType)
model_rebuild(PrismaDict)
model_rebuild(PrismaAlias)
