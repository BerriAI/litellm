from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polars._utils.unstable import issue_unstable_warning

if TYPE_CHECKING:
    from collections.abc import Collection

    from typing_extensions import TypeAlias


FloatCastOption: TypeAlias = Literal["upcast", "downcast"]
DatetimeCastOption: TypeAlias = Literal["nanosecond-downcast", "convert-timezone"]


class ScanCastOptions:
    """Options for scanning files."""

    def __init__(
        self,
        *,
        integer_cast: Literal["upcast", "forbid"] = "forbid",
        float_cast: Literal["forbid"]
        | FloatCastOption
        | Collection[FloatCastOption] = "forbid",
        datetime_cast: Literal["forbid"]
        | DatetimeCastOption
        | Collection[DatetimeCastOption] = "forbid",
        missing_struct_fields: Literal["insert", "raise"] = "raise",
        extra_struct_fields: Literal["ignore", "raise"] = "raise",
        _internal_call: bool = False,
    ) -> None:
        """
        Common configuration for scanning files.

        .. warning::
                This functionality is considered **unstable**. It may be changed
                at any point without it being considered a breaking change.

        Parameters
        ----------
        integer_cast
            Configuration for casting from integer types:

            * `upcast`: Allow lossless casting to wider integer types.
            * `forbid`: Raises an error if dtypes do not match.

        float_cast
            Configuration for casting from float types:

            * `upcast`: Allow casting to higher precision float types.
            * `downcast`: Allow casting to lower precision float types.
            * `forbid`: Raises an error if dtypes do not match.

        datetime_cast
            Configuration for casting from datetime types:

            * `nanosecond-downcast`: Allow nanosecond precision datetime to be \
            downcasted to any lower precision. This has a similar effect to \
            PyArrow's `coerce_int96_timestamp_unit`.
            * `convert-timezone`: Allow casting to a different timezone.
            * `forbid`: Raises an error if dtypes do not match.

        missing_struct_fields
            Configuration for behavior when struct fields defined in the schema
            are missing from the data:

            * `insert`: Inserts the missing fields.
            * `raise`: Raises an error.

        extra_struct_fields
            Configuration for behavior when extra struct fields outside of the
            defined schema are encountered in the data:

            * `ignore`: Silently ignores.
            * `raise`: Raises an error.

        """
        if not _internal_call:
            issue_unstable_warning("ScanCastOptions is considered unstable.")

        self.integer_cast = integer_cast
        self.float_cast = float_cast
        self.datetime_cast = datetime_cast
        self.missing_struct_fields = missing_struct_fields
        self.extra_struct_fields = extra_struct_fields

    @staticmethod
    def _default() -> ScanCastOptions:
        return ScanCastOptions(_internal_call=True)
