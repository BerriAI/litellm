import os
import sys
import enum
import textwrap
import importlib
from pathlib import Path
from keyword import iskeyword
from itertools import chain
from importlib import machinery, util as importlib_util
from importlib.abc import InspectLoader
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Iterable,
    NoReturn,
    Optional,
    List,
    Tuple,
    TypeVar,
    Union,
    Iterator,
    Dict,
    Type,
    cast,
)
from typing_extensions import Annotated

import click
import pydantic
from pydantic.fields import PrivateAttr

from .utils import Faker, Sampler, clean_multiline
from .. import config
from ..utils import DEBUG_GENERATOR, assert_never
from .._compat import (
    PYDANTIC_V2,
    BaseConfig,
    ConfigDict,
    BaseSettings,
    BaseSettingsConfig,
    PlainSerializer,
    GenericModel,
    Field as FieldInfo,
    root_validator,
    field_validator,
    cached_property,
    model_rebuild,
)
from .._constants import QUERY_BUILDER_ALIASES
from ..errors import UnsupportedListTypeError


__all__ = (
    'AnyData',
    'PythonData',
    'DefaultData',
    'GenericData',
)

_ModelT = TypeVar('_ModelT', bound=pydantic.BaseModel)

# NOTE: this does not represent all the data that is passed by prisma

ATOMIC_FIELD_TYPES = ['Int', 'BigInt', 'Float']

TYPE_MAPPING = {
    'String': '_str',
    'Bytes': "'fields.Base64'",
    'DateTime': 'datetime.datetime',
    'Boolean': '_bool',
    'Int': '_int',
    'Float': '_float',
    'BigInt': '_int',
    'Json': "'fields.Json'",
    'Decimal': 'decimal.Decimal',
}
FILTER_TYPES = [
    'String',
    'Bytes',
    'DateTime',
    'Boolean',
    'Int',
    'BigInt',
    'Float',
    'Json',
    'Decimal',
]
RECURSIVE_TYPE_DEPTH_WARNING = """Some types are disabled by default due to being incompatible with Mypy, it is highly recommended
to use Pyright instead and configure Prisma Python to use recursive types. To re-enable certain types:"""

RECURSIVE_TYPE_DEPTH_WARNING_DESC = """
generator client {
  provider             = "prisma-client-py"
  recursive_type_depth = -1
}

If you need to use Mypy, you can also disable this message by explicitly setting the default value:

generator client {
  provider             = "prisma-client-py"
  recursive_type_depth = 5
}

For more information see: https://prisma-client-py.readthedocs.io/en/stable/reference/limitations/#default-type-limitations
"""

FAKER: Faker = Faker()


ConfigT = TypeVar('ConfigT', bound=pydantic.BaseModel)

# Although we should just be able to access the config from the datamodel
# we have to do some validation that requires access to the config, this is difficult
# with heavily nested models as our current workaround only sets the datamodel context
# post-validation meaning we cannot access it in validators. To get around this we have
# a separate config context.
# TODO: better solution
data_ctx: ContextVar['AnyData'] = ContextVar('data_ctx')
config_ctx: ContextVar['Config'] = ContextVar('config_ctx')


def get_datamodel() -> 'Datamodel':
    return data_ctx.get().dmmf.datamodel


# typed to ensure the caller has to handle the cases where:
# - a custom generator config is being used
# - the config is invalid and therefore could not be set
def get_config() -> Union[None, pydantic.BaseModel, 'Config']:
    return config_ctx.get(None)


def get_list_types() -> Iterable[Tuple[str, str]]:
    # WARNING: do not edit this function without also editing Field.is_supported_scalar_list_type()
    return chain(
        ((t, TYPE_MAPPING[t]) for t in FILTER_TYPES),
        (
            (enum.name, f"'enums.{enum.name}'")
            for enum in get_datamodel().enums
        ),
    )


def sql_param(num: int = 1) -> str:
    # TODO: add case for sqlserver
    active_provider = data_ctx.get().datasources[0].active_provider
    if active_provider == 'postgresql':
        return f'${num}'

    # TODO: test
    if active_provider == 'mongodb':  # pragma: no cover
        raise RuntimeError('no-op')

    # SQLite and MySQL use this style so just default to it
    return '?'


def raise_err(msg: str) -> NoReturn:
    raise TemplateError(msg)


def type_as_string(typ: str) -> str:
    """Ensure a type string is wrapped with a string, e.g.

    enums.Role -> 'enums.Role'
    """
    # TODO: use this function internally in this module
    if not typ.startswith("'") and not typ.startswith('"'):
        return f"'{typ}'"
    return typ


def format_documentation(doc: str, indent: int = 4) -> str:
    """Format a schema comment by indenting nested lines, e.g.

        '''Foo
    Bar'''

    Becomes

        '''Foo
        Bar
        '''
    """
    if not doc:
        # empty string, nothing to do
        return doc

    prefix = ' ' * indent
    first, *rest = doc.splitlines()
    return '\n'.join(
        [
            first,
            *[textwrap.indent(line, prefix) for line in rest],
            prefix,
        ]
    )


def _module_spec_serializer(spec: machinery.ModuleSpec) -> str:
    assert spec.origin is not None, 'Cannot serialize module with no origin'
    return spec.origin


def _pathlib_serializer(path: Path) -> str:
    return str(path.absolute())


def _recursive_type_depth_factory() -> int:
    click.echo(
        click.style(
            f'\n{RECURSIVE_TYPE_DEPTH_WARNING}',
            fg='yellow',
        )
    )
    click.echo(f'{RECURSIVE_TYPE_DEPTH_WARNING_DESC}\n')
    return 5


class BaseModel(pydantic.BaseModel):
    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True,
            ignored_types=(cached_property,),
        )
    else:

        class Config(BaseConfig):
            arbitrary_types_allowed: bool = True
            json_encoders: Dict[Type[Any], Any] = {
                Path: _pathlib_serializer,
                machinery.ModuleSpec: _module_spec_serializer,
            }
            keep_untouched: Tuple[Type[Any], ...] = (cached_property,)


class InterfaceChoices(str, enum.Enum):
    sync = 'sync'
    asyncio = 'asyncio'


class EngineType(str, enum.Enum):
    binary = 'binary'
    library = 'library'
    dataproxy = 'dataproxy'

    def __str__(self) -> str:
        return self.value


class Module(BaseModel):
    if TYPE_CHECKING:
        spec: machinery.ModuleSpec
    else:
        if PYDANTIC_V2:
            spec: Annotated[
                machinery.ModuleSpec,
                PlainSerializer(
                    lambda x: _module_spec_serializer(x), return_type=str
                ),
            ]
        else:
            spec: machinery.ModuleSpec

    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True
        )
    else:

        class Config(BaseModel.Config):
            arbitrary_types_allowed: bool = True

    # for some reason this is needed in Pydantic v2
    @root_validator(pre=True, skip_on_failure=True)
    @classmethod
    def partial_type_generator_converter(cls, values: object) -> Any:
        if isinstance(values, str):
            return {'spec': values}
        return values

    @field_validator('spec', pre=True, allow_reuse=True)
    @classmethod
    def spec_validator(cls, value: Optional[str]) -> machinery.ModuleSpec:
        spec: Optional[machinery.ModuleSpec] = None

        # TODO: this should really work based off of the schema path
        # and this should suport checking  just partial_types.py if we are in a `prisma` dir
        if value is None:
            value = 'prisma/partial_types.py'

        path = Path.cwd().joinpath(value)
        if path.exists():
            spec = importlib_util.spec_from_file_location(
                'prisma.partial_type_generator', value
            )
        elif value.startswith('.'):
            raise ValueError(
                f'No file found at {value} and relative imports are not allowed.'
            )
        else:
            try:
                spec = importlib_util.find_spec(value)
            except ModuleNotFoundError:
                spec = None

        if spec is None:
            raise ValueError(
                f'Could not find a python file or module at {value}'
            )

        return spec

    def run(self) -> None:
        importlib.invalidate_caches()
        mod = importlib_util.module_from_spec(self.spec)
        loader = self.spec.loader
        assert loader is not None, 'Expected an import loader to exist.'
        assert isinstance(
            loader, InspectLoader
        ), f'Cannot execute module from loader type: {type(loader)}'

        try:
            loader.exec_module(mod)
        except Exception as exc:
            raise PartialTypeGeneratorError() from exc


class GenericData(GenericModel, Generic[ConfigT]):
    """Root model for the data that prisma provides to the generator.

    WARNING: only one instance of this class may exist at any given time and
    instances should only be constructed using the Data.parse_obj() method
    """

    datamodel: str
    version: str
    generator: 'Generator[ConfigT]'
    dmmf: 'DMMF' = FieldInfo(alias='dmmf')
    schema_path: Path = FieldInfo(alias='schemaPath')
    datasources: List['Datasource'] = FieldInfo(alias='datasources')
    other_generators: List['Generator[_ModelAllowAll]'] = FieldInfo(
        alias='otherGenerators'
    )
    binary_paths: 'BinaryPaths' = FieldInfo(
        alias='binaryPaths', default_factory=lambda: BinaryPaths()
    )

    if PYDANTIC_V2:

        @root_validator(pre=False)
        def _set_ctx(self: _ModelT) -> _ModelT:
            data_ctx.set(cast('GenericData[ConfigT]', self))
            return self

    else:

        @classmethod
        def parse_obj(cls, obj: Any) -> 'GenericData[ConfigT]':
            data = super().parse_obj(obj)  # pyright: ignore[reportDeprecated]
            data_ctx.set(data)
            return data

    def to_params(self) -> Dict[str, Any]:
        """Get the parameters that should be sent to Jinja templates"""
        params = vars(self)
        params['type_schema'] = Schema.from_data(self)

        # add utility functions
        for func in [
            sql_param,
            raise_err,
            type_as_string,
            get_list_types,
            clean_multiline,
            format_documentation,
        ]:
            params[func.__name__] = func

        return params

    @root_validator(pre=True, allow_reuse=True, skip_on_failure=True)
    @classmethod
    def validate_version(cls, values: Dict[Any, Any]) -> Dict[Any, Any]:
        # TODO: test this
        version = values.get('version')
        if not DEBUG_GENERATOR and version != config.expected_engine_version:
            raise ValueError(
                f'Prisma Client Python expected Prisma version: {config.expected_engine_version} '
                f'but got: {version}\n'
                '  If this is intentional, set the PRISMA_PY_DEBUG_GENERATOR environment '
                'variable to 1 and try again.\n'
                f'  If you are using the Node CLI then you must switch to v{config.prisma_version}, e.g. '
                f'npx prisma@{config.prisma_version} generate\n'
                '  or generate the client using the Python CLI, e.g. python3 -m prisma generate'
            )
        return values


class BinaryPaths(BaseModel):
    """This class represents the paths to engine binaries.

    Each property in this class is a mapping of platform name to absolute path, for example:

    ```py
    # This is what will be set on an M1 chip if there are no other `binaryTargets` set
    binary_paths.query_engine == {
        'darwin-arm64': '/Users/robert/.cache/prisma-python/binaries/3.13.0/efdf9b1183dddfd4258cd181a72125755215ab7b/node_modules/prisma/query-engine-darwin-arm64'
    }
    ```

    This is only available if the generator explicitly requests them using the `requires_engines` manifest property.
    """

    query_engine: Dict[str, str] = FieldInfo(
        default_factory=dict,
        alias='queryEngine',
    )
    introspection_engine: Dict[str, str] = FieldInfo(
        default_factory=dict,
        alias='introspectionEngine',
    )
    migration_engine: Dict[str, str] = FieldInfo(
        default_factory=dict,
        alias='migrationEngine',
    )
    libquery_engine: Dict[str, str] = FieldInfo(
        default_factory=dict,
        alias='libqueryEngine',
    )
    prisma_format: Dict[str, str] = FieldInfo(
        default_factory=dict,
        alias='prismaFmt',
    )

    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')
    else:

        class Config(BaseModel.Config):  # pyright: ignore[reportDeprecated]
            extra: Any = (
                pydantic.Extra.allow  # pyright: ignore[reportDeprecated]
            )


class Datasource(BaseModel):
    # TODO: provider enums
    name: str
    provider: str
    active_provider: str = FieldInfo(alias='activeProvider')
    url: 'OptionalValueFromEnvVar'


class Generator(GenericModel, Generic[ConfigT]):
    name: str
    output: 'ValueFromEnvVar'
    provider: 'OptionalValueFromEnvVar'
    config: ConfigT
    binary_targets: List['ValueFromEnvVar'] = FieldInfo(alias='binaryTargets')
    preview_features: List[str] = FieldInfo(alias='previewFeatures')

    @field_validator('binary_targets')
    @classmethod
    def warn_binary_targets(
        cls, targets: List['ValueFromEnvVar']
    ) -> List['ValueFromEnvVar']:
        # Prisma by default sends one binary target which is the current platform.
        if len(targets) > 1:
            click.echo(
                click.style(
                    'Warning: '
                    + 'The binaryTargets option is not officially supported by Prisma Client Python.',
                    fg='yellow',
                ),
                file=sys.stdout,
            )

        return targets

    def has_preview_feature(self, feature: str) -> bool:
        return feature in self.preview_features


class ValueFromEnvVar(BaseModel):
    value: str
    from_env_var: Optional[str] = FieldInfo(alias='fromEnvVar')


class OptionalValueFromEnvVar(BaseModel):
    value: Optional[str] = None
    from_env_var: Optional[str] = FieldInfo(alias='fromEnvVar')

    def resolve(self) -> str:
        value = self.value
        if value is not None:
            return value

        env_var = self.from_env_var
        assert env_var is not None, 'from_env_var should not be None'
        value = os.environ.get(env_var)
        if value is None:
            raise RuntimeError(f'Environment variable not found: {env_var}')

        return value


class Config(BaseSettings):
    """Custom generator config options."""

    interface: InterfaceChoices = FieldInfo(
        default=InterfaceChoices.asyncio, env='PRISMA_PY_CONFIG_INTERFACE'
    )
    partial_type_generator: Optional[Module] = FieldInfo(
        default=None, env='PRISMA_PY_CONFIG_PARTIAL_TYPE_GENERATOR'
    )
    recursive_type_depth: int = FieldInfo(
        default_factory=_recursive_type_depth_factory,
        env='PRISMA_PY_CONFIG_RECURSIVE_TYPE_DEPTH',
    )
    engine_type: EngineType = FieldInfo(
        default=EngineType.binary, env='PRISMA_PY_CONFIG_ENGINE_TYPE'
    )

    # this should be a list of experimental features
    # https://github.com/prisma/prisma/issues/12442
    enable_experimental_decimal: bool = FieldInfo(
        default=False, env='PRISMA_PY_CONFIG_ENABLE_EXPERIMENTAL_DECIMAL'
    )

    # this seems to be the only good method for setting the contextvar as
    # we don't control the actual construction of the object like we do for
    # the Data model.
    # we do not expose this to type checkers so that the generated __init__
    # signature is preserved.
    if not TYPE_CHECKING:

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            config_ctx.set(self)

    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(
            extra='forbid',
            use_enum_values=True,
            populate_by_name=True,
        )
    else:
        if not TYPE_CHECKING:

            class Config(BaseSettingsConfig):
                extra: pydantic.Extra = pydantic.Extra.forbid
                use_enum_values: bool = True
                env_prefix: str = 'prisma_py_config_'
                allow_population_by_field_name: bool = True

                @classmethod
                def customise_sources(
                    cls, init_settings, env_settings, file_secret_settings
                ):
                    # prioritise env settings over init settings
                    return env_settings, init_settings, file_secret_settings

    @root_validator(pre=True, skip_on_failure=True)
    @classmethod
    def transform_engine_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # prioritise env variable over schema option
        engine_type = os.environ.get('PRISMA_CLIENT_ENGINE_TYPE')
        if engine_type is None:
            engine_type = values.get('engineType')

        # only add engine_type if it is present
        if engine_type is not None:
            values['engine_type'] = engine_type
            values.pop('engineType', None)

        return values

    @root_validator(pre=True, skip_on_failure=True)
    @classmethod
    def removed_http_option_validator(
        cls, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        http = values.get('http')
        if http is not None:
            if http in {'aiohttp', 'httpx-async'}:
                option = 'asyncio'
            elif http in {'requests', 'httpx-sync'}:
                option = 'sync'
            else:  # pragma: no cover
                # invalid http option, let pydantic handle the error
                return values

            raise ValueError(
                'The http option has been removed in favour of the interface option.\n'
                '  Please remove the http option from your Prisma schema and replace it with:\n'
                f'  interface = "{option}"'
            )
        return values

    if PYDANTIC_V2:

        @root_validator(pre=True, skip_on_failure=True)
        @classmethod
        def partial_type_generator_converter(
            cls, values: Dict[str, Any]
        ) -> Dict[str, Any]:
            # ensure env resolving happens
            values = cast(Dict[str, Any], cls.root_validator(values))  # type: ignore

            value = values.get('partial_type_generator')

            try:
                values['partial_type_generator'] = Module(
                    spec=value  # pyright: ignore[reportGeneralTypeIssues]
                )
            except ValueError:
                if value is None:
                    # no config value passed and the default location was not found
                    return values
                raise

            return values

    else:

        @field_validator(
            'partial_type_generator', pre=True, always=True, allow_reuse=True
        )
        @classmethod
        def _partial_type_generator_converter(
            cls, value: Optional[str]
        ) -> Optional[Module]:
            try:
                return Module(
                    spec=value  # pyright: ignore[reportGeneralTypeIssues]
                )
            except ValueError:
                if value is None:
                    # no config value passed and the default location was not found
                    return None
                raise

    @field_validator('recursive_type_depth', always=True, allow_reuse=True)
    @classmethod
    def recursive_type_depth_validator(cls, value: int) -> int:
        if value < -1 or value in {0, 1}:
            raise ValueError('Value must equal -1 or be greater than 1.')
        return value

    @field_validator('engine_type', always=True, allow_reuse=True)
    @classmethod
    def engine_type_validator(cls, value: EngineType) -> EngineType:
        if value == EngineType.binary:
            return value
        elif value == EngineType.dataproxy:  # pragma: no cover
            raise ValueError(
                'Prisma Client Python does not support the Prisma Data Proxy yet.'
            )
        elif value == EngineType.library:  # pragma: no cover
            raise ValueError(
                'Prisma Client Python does not support native engine bindings yet.'
            )
        else:  # pragma: no cover
            assert_never(value)


class DMMF(BaseModel):
    datamodel: 'Datamodel'

    # TODO
    prisma_schema: Any = FieldInfo(alias='schema')


class Datamodel(BaseModel):
    enums: List['Enum']
    models: List['Model']

    # not implemented yet
    types: List[object]

    @field_validator('types')
    @classmethod
    def no_composite_types_validator(cls, types: List[object]) -> object:
        if types:
            raise ValueError(
                'Composite types are not supported yet. Please indicate you need this here: https://github.com/RobertCraigie/prisma-client-py/issues/314'
            )

        return types


class Enum(BaseModel):
    name: str
    db_name: Optional[str] = FieldInfo(alias='dbName')
    values: List['EnumValue']


class EnumValue(BaseModel):
    name: str
    db_name: Optional[str] = FieldInfo(alias='dbName')


class Model(BaseModel):
    name: str
    documentation: Optional[str] = None
    db_name: Optional[str] = FieldInfo(alias='dbName')
    is_generated: bool = FieldInfo(alias='isGenerated')
    compound_primary_key: Optional['PrimaryKey'] = FieldInfo(
        alias='primaryKey'
    )
    unique_indexes: List['UniqueIndex'] = FieldInfo(alias='uniqueIndexes')
    all_fields: List['Field'] = FieldInfo(alias='fields')

    _sampler: Sampler = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._sampler = Sampler(self)

    @field_validator('name')
    @classmethod
    def name_validator(cls, name: str) -> str:
        if iskeyword(name):
            raise ValueError(
                f'Model name "{name}" shadows a Python keyword; '
                f'use a different model name with \'@@map("{name}")\'.'
            )

        if iskeyword(name.lower()):
            raise ValueError(
                f'Model name "{name}" results in a client property that shadows a Python keyword; '
                f'use a different model name with \'@@map("{name}")\'.'
            )

        return name

    @property
    def related_models(self) -> Iterator['Model']:
        models = get_datamodel().models
        for field in self.relational_fields:
            for model in models:
                if field.type == model.name:
                    yield model

    @property
    def relational_fields(self) -> Iterator['Field']:
        for field in self.all_fields:
            if field.is_relational:
                yield field

    @property
    def scalar_fields(self) -> Iterator['Field']:
        for field in self.all_fields:
            if not field.is_relational:
                yield field

    @property
    def atomic_fields(self) -> Iterator['Field']:
        for field in self.all_fields:
            if field.type in ATOMIC_FIELD_TYPES:
                yield field

    @property
    def required_array_fields(self) -> Iterator['Field']:
        for field in self.all_fields:
            if field.is_list and not field.relation_name and field.is_required:
                yield field

    # TODO: support combined unique constraints
    @cached_property
    def id_field(self) -> Optional['Field']:
        """Find a field that can be passed to the model's `WhereUnique` filter"""
        for field in self.scalar_fields:  # pragma: no branch
            if field.is_id or field.is_unique:
                return field
        return None

    @property
    def has_relational_fields(self) -> bool:
        try:
            next(self.relational_fields)
        except StopIteration:
            return False
        else:
            return True

    @property
    def plural_name(self) -> str:
        name = self.name
        if name.endswith('s'):
            return name
        return f'{name}s'

    def resolve_field(self, name: str) -> 'Field':
        for field in self.all_fields:
            if field.name == name:
                return field

        raise LookupError(f'Could not find a field with name: {name}')

    def sampler(self) -> Sampler:
        return self._sampler


class Constraint(BaseModel):
    name: str
    fields: List[str]

    @root_validator(pre=True, allow_reuse=True, skip_on_failure=True)
    @classmethod
    def resolve_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        name = values.get('name')
        if isinstance(name, str):
            return values

        values['name'] = '_'.join(values['fields'])
        return values


class PrimaryKey(Constraint):
    pass


class UniqueIndex(Constraint):
    pass


class Field(BaseModel):
    name: str
    documentation: Optional[str] = None

    # TODO: switch to enums
    kind: str
    type: str

    is_id: bool = FieldInfo(alias='isId')
    is_list: bool = FieldInfo(alias='isList')
    is_unique: bool = FieldInfo(alias='isUnique')
    is_required: bool = FieldInfo(alias='isRequired')
    is_read_only: bool = FieldInfo(alias='isReadOnly')
    is_generated: bool = FieldInfo(alias='isGenerated')
    is_updated_at: bool = FieldInfo(alias='isUpdatedAt')

    default: Optional[Union['DefaultValue', object, List[object]]] = None
    has_default_value: bool = FieldInfo(alias='hasDefaultValue')

    relation_name: Optional[str] = FieldInfo(
        alias='relationName', default=None
    )
    relation_on_delete: Optional[str] = FieldInfo(
        alias='relationOnDelete', default=None
    )
    relation_to_fields: Optional[List[str]] = FieldInfo(
        alias='relationToFields',
        default=None,
    )
    relation_from_fields: Optional[List[str]] = FieldInfo(
        alias='relationFromFields',
        default=None,
    )

    _last_sampled: Optional[str] = PrivateAttr()

    @root_validator(pre=True, skip_on_failure=True)
    @classmethod
    def scalar_type_validator(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        kind = values.get('kind')
        type_ = values.get('type')

        if kind == 'scalar':
            if type_ is not None and type_ not in TYPE_MAPPING:
                raise ValueError(f'Unsupported scalar field type: {type_}')

        return values

    @field_validator('type')
    @classmethod
    def experimental_decimal_validator(cls, typ: str) -> str:
        if typ == 'Decimal':
            config = get_config()

            # skip validating the experimental flag if we are
            # being called from a custom generator
            if (
                isinstance(config, Config)
                and not config.enable_experimental_decimal
            ):
                raise ValueError(
                    'Support for the Decimal type is experimental\n'
                    '  As such you must set the `enable_experimental_decimal` config flag to true\n'
                    '  for more information see: https://github.com/RobertCraigie/prisma-client-py/issues/106'
                )

        return typ

    @field_validator('name')
    @classmethod
    def name_validator(cls, name: str) -> str:
        if getattr(BaseModel, name, None):
            raise ValueError(
                f'Field name "{name}" shadows a BaseModel attribute; '
                f'use a different field name with \'@map("{name}")\'.'
            )

        if iskeyword(name):
            raise ValueError(
                f'Field name "{name}" shadows a Python keyword; '
                f'use a different field name with \'@map("{name}")\'.'
            )

        if name == 'prisma':
            raise ValueError(
                'Field name "prisma" shadows a Prisma Client Python method; '
                'use a different field name with \'@map("prisma")\'.'
            )

        if name in QUERY_BUILDER_ALIASES:
            raise ValueError(
                f'Field name "{name}" shadows an internal keyword; '
                f'use a different field name with \'@map("{name}")\''
            )

        return name

    # TODO: cache the properties
    @property
    def python_type(self) -> str:
        type_ = self._actual_python_type
        if self.is_list:
            return f'List[{type_}]'
        return type_

    @property
    def python_type_as_string(self) -> str:
        type_ = self._actual_python_type
        if self.is_list:
            type_ = type_.replace("'", "\\'")
            return f"'List[{type_}]'"

        if not type_.startswith("'"):
            type_ = f"'{type_}'"

        return type_

    @property
    def _actual_python_type(self) -> str:
        if self.kind == 'enum':
            return f"'enums.{self.type}'"

        if self.kind == 'object':
            return f"'models.{self.type}'"

        try:
            return TYPE_MAPPING[self.type]
        except KeyError as exc:
            # TODO: handle this better
            raise RuntimeError(
                f'Could not parse {self.name} due to unknown type: {self.type}',
            ) from exc

    @property
    def create_input_type(self) -> str:
        if self.kind != 'object':
            return self.python_type

        if self.is_list:
            return f"'{self.type}CreateManyNestedWithoutRelationsInput'"

        return f"'{self.type}CreateNestedWithoutRelationsInput'"

    @property
    def where_input_type(self) -> str:
        typ = self.type
        if self.is_relational:
            if self.is_list:
                return f"'{typ}ListRelationFilter'"
            return f"'{typ}RelationFilter'"

        if self.is_list:
            self.check_supported_scalar_list_type()
            return f"'types.{typ}ListFilter'"

        if typ in FILTER_TYPES:
            if self.is_optional:
                return f"Union[None, {self._actual_python_type}, 'types.{typ}Filter']"
            return f"Union[{self._actual_python_type}, 'types.{typ}Filter']"

        return self.python_type

    @property
    def where_aggregates_input_type(self) -> str:
        if self.is_relational:  # pragma: no cover
            raise RuntimeError('This type is not valid for relational fields')

        typ = self.type
        if typ in FILTER_TYPES:
            return f"Union[{self._actual_python_type}, 'types.{typ}WithAggregatesFilter']"
        return self.python_type

    @property
    def relational_args_type(self) -> str:
        if self.is_list:
            return f'FindMany{self.type}Args'
        return f'{self.type}Args'

    @property
    def required_on_create(self) -> bool:
        return (
            self.is_required
            and not self.is_updated_at
            and not self.has_default_value
            and not self.relation_name
            and not self.is_list
        )

    @property
    def is_optional(self) -> bool:
        return not (self.is_required and not self.relation_name)

    @property
    def is_relational(self) -> bool:
        return self.relation_name is not None

    @property
    def is_atomic(self) -> bool:
        return self.type in ATOMIC_FIELD_TYPES

    @property
    def is_number(self) -> bool:
        return self.type in {'Int', 'BigInt', 'Float'}

    def maybe_optional(self, typ: str) -> str:
        """Wrap the given type string within `Optional` if applicable"""
        if self.is_required or self.is_relational:
            return typ
        return f'Optional[{typ}]'

    def get_update_input_type(self) -> str:
        if self.kind == 'object':
            if self.is_list:
                return f"'{self.type}UpdateManyWithoutRelationsInput'"
            return f"'{self.type}UpdateOneWithoutRelationsInput'"

        if self.is_list:
            self.check_supported_scalar_list_type()
            return f"'types.{self.type}ListUpdate'"

        if self.is_atomic:
            return f'Union[Atomic{self.type}Input, {self.python_type}]'

        return self.python_type

    def check_supported_scalar_list_type(self) -> None:
        if (
            self.type not in FILTER_TYPES and self.kind != 'enum'
        ):  # pragma: no branch
            raise UnsupportedListTypeError(self.type)

    def get_relational_model(self) -> Optional['Model']:
        if not self.is_relational:
            return None

        name = self.type
        for model in get_datamodel().models:
            if model.name == name:
                return model
        return None

    def get_corresponding_enum(self) -> Optional['Enum']:
        typ = self.type
        for enum in get_datamodel().enums:
            if enum.name == typ:
                return enum
        return None  # pragma: no cover

    def get_sample_data(self, *, increment: bool = True) -> str:
        # returning the same data that was last sampled is useful
        # for documenting methods like upsert() where data is duplicated
        if not increment and self._last_sampled is not None:
            return self._last_sampled

        sampled = self._get_sample_data()
        if self.is_list:
            sampled = f'[{sampled}]'

        self._last_sampled = sampled
        return sampled

    def _get_sample_data(self) -> str:
        if self.is_relational:  # pragma: no cover
            raise RuntimeError(
                'Data sampling for relational fields not supported yet'
            )

        if self.kind == 'enum':
            enum = self.get_corresponding_enum()
            assert enum is not None, self.type
            return f'enums.{enum.name}.{FAKER.from_list(enum.values).name}'

        typ = self.type
        if typ == 'Boolean':
            return str(FAKER.boolean())
        elif typ == 'Int':
            return str(FAKER.integer())
        elif typ == 'String':
            return f"'{FAKER.string()}'"
        elif typ == 'Float':
            return f'{FAKER.integer()}.{FAKER.integer() // 10000}'
        elif typ == 'BigInt':  # pragma: no cover
            return str(FAKER.integer() * 12)
        elif typ == 'DateTime':
            # TODO: random dates
            return 'datetime.datetime.utcnow()'
        elif typ == 'Json':
            return f"Json({{'{FAKER.string()}': True}})"
        elif typ == 'Bytes':
            return f"Base64.encode(b'{FAKER.string()}')"
        elif typ == 'Decimal':
            return f"Decimal('{FAKER.integer()}.{FAKER.integer() // 10000}')"
        else:  # pragma: no cover
            raise RuntimeError(f'Sample data not supported for {typ} yet')


class DefaultValue(BaseModel):
    args: Any = None
    name: str


class _EmptyModel(BaseModel):
    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid')
    elif not TYPE_CHECKING:

        class Config(BaseModel.Config):
            extra: pydantic.Extra = pydantic.Extra.forbid


class _ModelAllowAll(BaseModel):
    if PYDANTIC_V2:
        model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow')
    elif not TYPE_CHECKING:

        class Config(BaseModel.Config):
            extra: pydantic.Extra = pydantic.Extra.allow


class PythonData(GenericData[Config]):
    """Data class including the default Prisma Client Python config"""

    if not PYDANTIC_V2:

        class Config(BaseConfig):
            arbitrary_types_allowed: bool = True
            json_encoders: Dict[Type[Any], Any] = {
                Path: _pathlib_serializer,
                machinery.ModuleSpec: _module_spec_serializer,
            }
            keep_untouched: Tuple[Type[Any], ...] = (cached_property,)


class DefaultData(GenericData[_EmptyModel]):
    """Data class without any config options"""


# this has to be defined as a type alias instead of a class
# as its purpose is to signify that the data is config agnostic
AnyData = GenericData[Any]

model_rebuild(Enum)
model_rebuild(DMMF)
model_rebuild(GenericData)
model_rebuild(Field)
model_rebuild(Model)
model_rebuild(Datamodel)
model_rebuild(Generator)
model_rebuild(Datasource)


from .schema import Schema
from .errors import (
    PartialTypeGeneratorError,
    TemplateError,
)
