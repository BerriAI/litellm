from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars._reexport as pl

if TYPE_CHECKING:
    import contextlib

    with contextlib.suppress(ImportError):  # Module not available when building docs
        from polars.polars import PyDataTypeExpr
    from polars import DataType
    from polars._typing import SchemaDict


class DataTypeExpr:
    """
    A lazily instantiated :class:`DataType` that can be used in an :class:`Expr`.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    A expression made to represent a :class:`DataType` that can be used to
    reference the datatype of an expression of column in a lazy context.

    Examples
    --------
    >>> lf = pl.LazyFrame({"a": [1, 2, 3]})
    >>> lf.with_columns(
    ...     pl.col.a.map_batches(lambda x: x * 2, return_dtype=pl.dtype_of("a"))
    ... ).collect()
    shape: (3, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 2   │
    │ 4   │
    │ 6   │
    └─────┘
    """

    _pydatatype_expr: PyDataTypeExpr

    @classmethod
    def _from_pydatatype_expr(cls, pydatatype_expr: PyDataTypeExpr) -> DataTypeExpr:
        slf = cls()
        slf._pydatatype_expr = pydatatype_expr
        return slf

    def collect_dtype(
        self, context: SchemaDict | pl.Schema | pl.DataFrame | pl.LazyFrame
    ) -> DataType:
        """Materialize the :class:`DataTypeExpr` in a specific context."""
        schema: pl.Schema
        if isinstance(context, pl.Schema):
            schema = context
        elif isinstance(context, Mapping):
            schema = pl.Schema(context)
        elif isinstance(context, pl.DataFrame):
            schema = context.schema
        elif isinstance(context, pl.LazyFrame):
            schema = context.collect_schema()
        else:
            msg = f"DataTypeExpr.collect_dtype did not expect {context!r}"
            raise TypeError(msg)

        return self._pydatatype_expr.collect_dtype(schema)
