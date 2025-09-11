"""Re-export Polars functionality to avoid cyclical imports."""

from polars.dataframe import DataFrame
from polars.datatype_expr import DataTypeExpr
from polars.expr import Expr, When
from polars.lazyframe import LazyFrame
from polars.schema import Schema
from polars.series import Series

__all__ = [
    "DataFrame",
    "DataTypeExpr",
    "Expr",
    "LazyFrame",
    "Schema",
    "Series",
    "When",
]
