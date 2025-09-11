from __future__ import annotations as _annotations

import os
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic._internal._utils import deep_update, is_model_class
from pydantic.dataclasses import is_pydantic_dataclass
from pydantic.fields import FieldInfo
from typing_extensions import get_args, get_origin
from typing_inspection.introspection import is_union_origin

from ...utils import _lenient_issubclass
from ..base import PydanticBaseEnvSettingsSource
from ..types import EnvNoneType
from ..utils import (
    _annotation_enum_name_to_val,
    _get_model_fields,
    _union_is_complex,
    parse_env_vars,
)

if TYPE_CHECKING:
    from pydantic_settings.main import BaseSettings


class EnvSettingsSource(PydanticBaseEnvSettingsSource):
    """
    Source class for loading settings values from environment variables.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        case_sensitive: bool | None = None,
        env_prefix: str | None = None,
        env_nested_delimiter: str | None = None,
        env_nested_max_split: int | None = None,
        env_ignore_empty: bool | None = None,
        env_parse_none_str: str | None = None,
        env_parse_enums: bool | None = None,
    ) -> None:
        super().__init__(
            settings_cls,
            case_sensitive,
            env_prefix,
            env_ignore_empty,
            env_parse_none_str,
            env_parse_enums,
        )
        self.env_nested_delimiter = (
            env_nested_delimiter
            if env_nested_delimiter is not None
            else self.config.get("env_nested_delimiter")
        )
        self.env_nested_max_split = (
            env_nested_max_split
            if env_nested_max_split is not None
            else self.config.get("env_nested_max_split")
        )
        self.maxsplit = (self.env_nested_max_split or 0) - 1
        self.env_prefix_len = len(self.env_prefix)

        self.env_vars = self._load_env_vars()

    def _load_env_vars(self) -> Mapping[str, str | None]:
        return parse_env_vars(
            os.environ,
            self.case_sensitive,
            self.env_ignore_empty,
            self.env_parse_none_str,
        )

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """
        Gets the value for field from environment variables and a flag to determine whether value is complex.

        Args:
            field: The field.
            field_name: The field name.

        Returns:
            A tuple that contains the value (`None` if not found), key, and
                a flag to determine whether value is complex.
        """

        env_val: str | None = None
        for field_key, env_name, value_is_complex in self._extract_field_info(
            field, field_name
        ):
            env_val = self.env_vars.get(env_name)
            if env_val is not None:
                break

        return env_val, field_key, value_is_complex

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        """
        Prepare value for the field.

        * Extract value for nested field.
        * Deserialize value to python object for complex field.

        Args:
            field: The field.
            field_name: The field name.

        Returns:
            A tuple contains prepared value for the field.

        Raises:
            ValuesError: When There is an error in deserializing value for complex field.
        """
        is_complex, allow_parse_failure = self._field_is_complex(field)
        if self.env_parse_enums:
            enum_val = _annotation_enum_name_to_val(field.annotation, value)
            value = value if enum_val is None else enum_val

        if is_complex or value_is_complex:
            if isinstance(value, EnvNoneType):
                return value
            elif value is None:
                # field is complex but no value found so far, try explode_env_vars
                env_val_built = self.explode_env_vars(field_name, field, self.env_vars)
                if env_val_built:
                    return env_val_built
            else:
                # field is complex and there's a value, decode that as JSON, then add explode_env_vars
                try:
                    value = self.decode_complex_value(field_name, field, value)
                except ValueError as e:
                    if not allow_parse_failure:
                        raise e

                if isinstance(value, dict):
                    return deep_update(
                        value, self.explode_env_vars(field_name, field, self.env_vars)
                    )
                else:
                    return value
        elif value is not None:
            # simplest case, field is not complex, we only need to add the value if it was found
            return value

    def _field_is_complex(self, field: FieldInfo) -> tuple[bool, bool]:
        """
        Find out if a field is complex, and if so whether JSON errors should be ignored
        """
        if self.field_is_complex(field):
            allow_parse_failure = False
        elif is_union_origin(get_origin(field.annotation)) and _union_is_complex(
            field.annotation, field.metadata
        ):
            allow_parse_failure = True
        else:
            return False, False

        return True, allow_parse_failure

    # Default value of `case_sensitive` is `None`, because we don't want to break existing behavior.
    # We have to change the method to a non-static method and use
    # `self.case_sensitive` instead in V3.
    def next_field(
        self,
        field: FieldInfo | Any | None,
        key: str,
        case_sensitive: bool | None = None,
    ) -> FieldInfo | None:
        """
        Find the field in a sub model by key(env name)

        By having the following models:

            ```py
            class SubSubModel(BaseSettings):
                dvals: Dict

            class SubModel(BaseSettings):
                vals: list[str]
                sub_sub_model: SubSubModel

            class Cfg(BaseSettings):
                sub_model: SubModel
            ```

        Then:
            next_field(sub_model, 'vals') Returns the `vals` field of `SubModel` class
            next_field(sub_model, 'sub_sub_model') Returns `sub_sub_model` field of `SubModel` class

        Args:
            field: The field.
            key: The key (env name).
            case_sensitive: Whether to search for key case sensitively.

        Returns:
            Field if it finds the next field otherwise `None`.
        """
        if not field:
            return None

        annotation = field.annotation if isinstance(field, FieldInfo) else field
        for type_ in get_args(annotation):
            type_has_key = self.next_field(type_, key, case_sensitive)
            if type_has_key:
                return type_has_key
        if is_model_class(annotation) or is_pydantic_dataclass(annotation):  # type: ignore[arg-type]
            fields = _get_model_fields(annotation)
            # `case_sensitive is None` is here to be compatible with the old behavior.
            # Has to be removed in V3.
            for field_name, f in fields.items():
                for _, env_name, _ in self._extract_field_info(f, field_name):
                    if case_sensitive is None or case_sensitive:
                        if field_name == key or env_name == key:
                            return f
                    elif (
                        field_name.lower() == key.lower()
                        or env_name.lower() == key.lower()
                    ):
                        return f
        return None

    def explode_env_vars(
        self, field_name: str, field: FieldInfo, env_vars: Mapping[str, str | None]
    ) -> dict[str, Any]:
        """
        Process env_vars and extract the values of keys containing env_nested_delimiter into nested dictionaries.

        This is applied to a single field, hence filtering by env_var prefix.

        Args:
            field_name: The field name.
            field: The field.
            env_vars: Environment variables.

        Returns:
            A dictionary contains extracted values from nested env values.
        """
        if not self.env_nested_delimiter:
            return {}

        ann = field.annotation
        is_dict = ann is dict or _lenient_issubclass(get_origin(ann), dict)

        prefixes = [
            f"{env_name}{self.env_nested_delimiter}"
            for _, env_name, _ in self._extract_field_info(field, field_name)
        ]
        result: dict[str, Any] = {}
        for env_name, env_val in env_vars.items():
            try:
                prefix = next(
                    prefix for prefix in prefixes if env_name.startswith(prefix)
                )
            except StopIteration:
                continue
            # we remove the prefix before splitting in case the prefix has characters in common with the delimiter
            env_name_without_prefix = env_name[len(prefix) :]
            *keys, last_key = env_name_without_prefix.split(
                self.env_nested_delimiter, self.maxsplit
            )
            env_var = result
            target_field: FieldInfo | None = field
            for key in keys:
                target_field = self.next_field(target_field, key, self.case_sensitive)
                if isinstance(env_var, dict):
                    env_var = env_var.setdefault(key, {})

            # get proper field with last_key
            target_field = self.next_field(target_field, last_key, self.case_sensitive)

            # check if env_val maps to a complex field and if so, parse the env_val
            if (target_field or is_dict) and env_val:
                if target_field:
                    is_complex, allow_json_failure = self._field_is_complex(
                        target_field
                    )
                    if self.env_parse_enums:
                        enum_val = _annotation_enum_name_to_val(
                            target_field.annotation, env_val
                        )
                        env_val = env_val if enum_val is None else enum_val
                else:
                    # nested field type is dict
                    is_complex, allow_json_failure = True, True
                if is_complex:
                    try:
                        env_val = self.decode_complex_value(last_key, target_field, env_val)  # type: ignore
                    except ValueError as e:
                        if not allow_json_failure:
                            raise e
            if isinstance(env_var, dict):
                if (
                    last_key not in env_var
                    or not isinstance(env_val, EnvNoneType)
                    or env_var[last_key] == {}
                ):
                    env_var[last_key] = env_val

        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env_nested_delimiter={self.env_nested_delimiter!r}, "
            f"env_prefix_len={self.env_prefix_len!r})"
        )


__all__ = ["EnvSettingsSource"]
