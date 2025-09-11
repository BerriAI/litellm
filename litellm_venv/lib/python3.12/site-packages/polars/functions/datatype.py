from __future__ import annotations

from typing import TYPE_CHECKING

import polars._reexport as pl
from polars import functions as F
from polars._utils.unstable import unstable

if TYPE_CHECKING:
    from polars import Expr


@unstable()
def dtype_of(col_or_expr: str | Expr) -> pl.DataTypeExpr:
    """
    Get a lazily evaluated :class:`DataType` of a column or expression.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """
    from polars.polars import PyDataTypeExpr

    e: Expr
    if isinstance(col_or_expr, str):
        e = F.col(col_or_expr)
    else:
        e = col_or_expr

    return pl.DataTypeExpr._from_pydatatype_expr(PyDataTypeExpr.of_expr(e._pyexpr))
