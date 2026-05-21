"""Prisma-client-py compatibility shim backed by SQLAlchemy.

The existing LiteLLM proxy has ~1,680 call sites of the form::

    await prisma_client.db.litellm_verificationtoken.find_unique(
        where={"token": hashed_token},
        include={"litellm_budget_table": True},
    )

Rewriting every call site at once is the diff size that blocks any
big-bang migration. Instead, this module exposes the same
``db.litellm_<table>.<method>(...)`` surface and translates each call
into a SQLAlchemy statement. Call sites continue to work unchanged.

What is implemented (and tested in the unit suite):

* per-table accessors auto-derived from :data:`models.ALL_MODELS`
* ``find_unique``, ``find_first``, ``find_many``, ``count``,
  ``create``, ``update``, ``upsert``, ``delete``,
  ``update_many``, ``delete_many``, ``create_many``
* ``where=`` translation for ``in``, ``not``, ``equals``, ``contains``
  (with ``mode: 'insensitive'``), ``startswith``, ``endswith``,
  ``gt``, ``lt``, ``gte``, ``lte``, ``has``, ``AND``, ``OR``, ``NOT``
* ``data=`` translation including ``{"increment": N}``
* ``order=``, ``take=``, ``skip=``
* ``query_raw(sql, *params)`` and ``query_raw(query=sql, *params)``
* ``execute_raw(sql, *params)``
* ``batch_()`` returning a batcher with ``await batcher.commit()``
* ``async with db.tx() as tx:``

What is **not** implemented (raises ``NotImplementedError`` or logs a
warning and returns ``None``):

* ``include=`` for relation eager-loading -- SQLModel classes do not
  carry SQLAlchemy ``relationship()`` definitions yet. Existing call
  sites that rely on ``include`` will see ``None`` for the included
  attribute. Adding ``relationship()`` declarations and selectinload
  wiring is a follow-up of comparable scope to a CRUD-feature port.
* ``select=`` for column projection -- only 3 production call sites
  use it; they should be ported to direct attribute access.
* ``group_by`` -- 4 production call sites; port to raw SQL.
* Nested ``some``/``every`` relation filters -- 2 production call
  sites; port to explicit JOIN via raw SQL.

The tests under ``tests/test_litellm/proxy/db/sqlmodel_orm/`` exercise
the implemented features against an in-memory SQLite DB.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
)

from sqlalchemy import (
    and_,
    delete,
    func,
    not_,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from litellm.proxy.db.sqlmodel import errors
from litellm.proxy.db.sqlmodel.engine import LiteLLMDB
from litellm.proxy.db.sqlmodel.models import ALL_MODELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table registry
# ---------------------------------------------------------------------------


def _build_table_index() -> Dict[str, Type[SQLModel]]:
    """Map ``litellm_<lowercased_modelname>`` -> SQLModel class.

    Mirrors prisma-client-py's accessor naming: the Prisma model
    ``LiteLLM_TeamTable`` becomes ``litellm_teamtable``.
    """
    index: Dict[str, Type[SQLModel]] = {}
    for cls in ALL_MODELS:
        # prisma-client-py exposes a model accessor by lowercasing the
        # original Prisma model name verbatim -- underscores included.
        # ``LiteLLM_VerificationToken`` -> ``litellm_verificationtoken``
        # ``LiteLLM_Config``            -> ``litellm_config``
        prisma_table = getattr(cls, "__tablename__", cls.__name__)
        accessor = prisma_table.lower()
        index[accessor] = cls
    return index


_TABLE_INDEX: Dict[str, Type[SQLModel]] = _build_table_index()


def _model_for_accessor(name: str) -> Type[SQLModel]:
    cls = _TABLE_INDEX.get(name)
    if cls is None:
        raise errors.TableNotFoundError(
            f"No SQLModel class registered for prisma accessor '{name}'. "
            "Did you regenerate models.py after a schema change?"
        )
    return cls


# ---------------------------------------------------------------------------
# Filter / data translators
# ---------------------------------------------------------------------------


_FIELD_OPERATORS = {
    "equals",
    "not",
    "in",
    "notIn",
    "not_in",
    "contains",
    "startswith",
    "endswith",
    "lt",
    "lte",
    "gt",
    "gte",
    "has",
    "mode",
}


def _resolve_attr(model: Type[SQLModel], name: str) -> Any:
    """Return the SQLAlchemy column for ``name`` on ``model``.

    Honours the Python-attribute-rename trick the generator uses for
    SQLAlchemy reserved names (``metadata`` -> ``metadata_``). Callers
    pass the on-disk column name (``metadata``); we look it up in
    ``__table__.columns`` first, then fall back to attribute lookup.
    """
    table = model.__table__  # type: ignore[attr-defined]
    if name in table.columns:
        return table.columns[name]
    attr = getattr(model, name, None)
    if attr is None:
        raise AttributeError(
            f"Column '{name}' not found on {model.__name__}; "
            "shim cannot translate this filter."
        )
    return attr


def _translate_field_filter(column: Any, operand: Any) -> Any:
    if not isinstance(operand, Mapping):
        return column == operand

    mode = operand.get("mode")
    case_insensitive = mode == "insensitive"
    clauses: List[Any] = []
    for op, val in operand.items():
        if op == "mode":
            continue
        if op == "equals":
            if case_insensitive and isinstance(val, str):
                clauses.append(func.lower(column) == val.lower())
            else:
                clauses.append(column == val)
        elif op == "not":
            clauses.append(column != val)
        elif op == "in":
            clauses.append(column.in_(list(val) if val is not None else []))
        elif op in ("notIn", "not_in"):
            clauses.append(~column.in_(list(val) if val is not None else []))
        elif op == "contains":
            pattern = f"%{val}%"
            clauses.append(
                column.ilike(pattern) if case_insensitive else column.like(pattern)
            )
        elif op == "startswith":
            pattern = f"{val}%"
            clauses.append(
                column.ilike(pattern) if case_insensitive else column.like(pattern)
            )
        elif op == "endswith":
            pattern = f"%{val}"
            clauses.append(
                column.ilike(pattern) if case_insensitive else column.like(pattern)
            )
        elif op == "lt":
            clauses.append(column < val)
        elif op == "lte":
            clauses.append(column <= val)
        elif op == "gt":
            clauses.append(column > val)
        elif op == "gte":
            clauses.append(column >= val)
        elif op == "has":
            # Postgres array contains -- ``column @> ARRAY[val]``.
            clauses.append(column.contains([val]))
        else:
            raise NotImplementedError(
                f"Field operator '{op}' not implemented in compat shim. "
                "Either port the call site to raw SQL or extend "
                "_translate_field_filter."
            )
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return and_(*clauses)


def _translate_where(model: Type[SQLModel], where: Optional[Mapping[str, Any]]) -> Any:
    """Translate a Prisma ``where=`` mapping to a SQLAlchemy clause.

    Returns ``None`` when ``where`` is empty (caller should emit no
    WHERE clause).
    """
    if not where:
        return None
    clauses: List[Any] = []
    for key, val in where.items():
        if key == "AND":
            assert isinstance(val, list), "AND requires a list of sub-filters"
            sub = [_translate_where(model, item) for item in val]
            clauses.append(and_(*[c for c in sub if c is not None]))
        elif key == "OR":
            assert isinstance(val, list), "OR requires a list of sub-filters"
            sub = [_translate_where(model, item) for item in val]
            clauses.append(or_(*[c for c in sub if c is not None]))
        elif key == "NOT":
            if isinstance(val, list):
                sub = [_translate_where(model, item) for item in val]
                clauses.append(not_(and_(*[c for c in sub if c is not None])))
            else:
                inner = _translate_where(model, val)
                if inner is not None:
                    clauses.append(not_(inner))
        elif key in ("some", "every", "is", "isNot", "is_not"):
            raise NotImplementedError(
                f"Relation filter '{key}' is not yet supported by the shim. "
                "Port this call site to an explicit JOIN via raw SQL."
            )
        else:
            column = _resolve_attr(model, key)
            translated = _translate_field_filter(column, val)
            if translated is not None:
                clauses.append(translated)
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return and_(*clauses)


def _translate_data(model: Type[SQLModel], data: Mapping[str, Any]) -> Dict[str, Any]:
    """Prisma ``data=`` -> kwargs/values for ``insert``/``update`` statements.

    Handles ``{"increment": N}`` by emitting a SQL expression; everything
    else passes through verbatim.
    """
    out: Dict[str, Any] = {}
    table = model.__table__  # type: ignore[attr-defined]
    for key, val in data.items():
        if isinstance(val, Mapping) and "increment" in val:
            column = table.columns[key]
            out[key] = column + val["increment"]
        elif isinstance(val, Mapping) and "decrement" in val:
            column = table.columns[key]
            out[key] = column - val["decrement"]
        elif isinstance(val, Mapping) and "set" in val:
            out[key] = val["set"]
        else:
            out[key] = val
    return out


def _translate_order(model: Type[SQLModel], order: Any) -> List[Any]:
    """Prisma ``order=`` -> list of SQLAlchemy ORDER BY expressions."""
    if order is None:
        return []
    if isinstance(order, Mapping):
        items = list(order.items())
    elif isinstance(order, list):
        items = []
        for entry in order:
            if isinstance(entry, Mapping):
                items.extend(entry.items())
        # Keep deterministic iteration order
    else:
        return []
    out: List[Any] = []
    for col_name, direction in items:
        column = _resolve_attr(model, col_name)
        if str(direction).lower().startswith("desc"):
            out.append(column.desc())
        else:
            out.append(column.asc())
    return out


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


class CountResult:
    """Mimic prisma-client-py's ``BatchPayload`` for ``update_many`` etc.

    Some call sites read ``result.count`` and others compare to an int;
    we support both by also implementing ``__int__`` and ``__index__``.
    """

    __slots__ = ("count",)

    def __init__(self, count: int) -> None:
        self.count = count

    def __int__(self) -> int:
        return self.count

    def __index__(self) -> int:
        return self.count

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CountResult):
            return self.count == other.count
        if isinstance(other, int):
            return self.count == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"CountResult(count={self.count})"


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    if isinstance(row, dict):
        return row
    return dict(row)  # last resort


# ---------------------------------------------------------------------------
# Per-table accessor
# ---------------------------------------------------------------------------


def _warn_unsupported(kwarg: str, table: str) -> None:
    if kwarg in ("include",):
        logger.warning(
            "compat shim: '%s=' on %s is not yet implemented; the relation "
            "attribute will be unset on returned rows.",
            kwarg,
            table,
        )
    elif kwarg in ("select",):
        logger.warning(
            "compat shim: 'select=' on %s is not yet implemented; returning "
            "the full row.",
            table,
        )


class TableAccessor:
    """The thing returned by ``client.db.litellm_<table>``.

    Two construction modes:

    * **standalone** (``owned_session=False``): each method opens its own
      session, performs the work, commits, and closes. This matches the
      historical Prisma-client-py call pattern where each ``await
      db.foo.bar()`` is its own implicit transaction.
    * **bound** (``owned_session=True``): the accessor reuses an existing
      :class:`AsyncSession` opened by a surrounding ``tx()`` context. The
      session is *not* closed and *not* committed -- the surrounding
      transaction owns lifetime.
    """

    def __init__(
        self,
        model: Type[SQLModel],
        db: "LiteLLMDB | None" = None,
        bound_session: "AsyncSession | None" = None,
    ) -> None:
        if (db is None) == (bound_session is None):
            raise ValueError(
                "TableAccessor requires exactly one of db= or bound_session="
            )
        self._model = model
        self._db = db
        self._bound = bound_session

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        if self._bound is not None:
            yield self._bound
            return
        assert self._db is not None
        async with self._db.session_ctx() as sess:
            yield sess

    @property
    def _is_bound(self) -> bool:
        return self._bound is not None

    async def _maybe_commit(self, sess: AsyncSession) -> None:
        if not self._is_bound:
            await sess.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def find_unique(
        self,
        *,
        where: Optional[Mapping[str, Any]] = None,
        include: Any = None,
        select: Any = None,  # noqa: A002 -- prisma kwarg name
    ) -> Optional[SQLModel]:
        if include is not None:
            _warn_unsupported("include", self._model.__tablename__)
        if select is not None:
            _warn_unsupported("select", self._model.__tablename__)
        clause = _translate_where(self._model, where)
        stmt = self._select_stmt().limit(1)
        if clause is not None:
            stmt = stmt.where(clause)
        return await self._scalar_one_or_none(stmt)

    async def find_first(
        self,
        *,
        where: Optional[Mapping[str, Any]] = None,
        order: Any = None,
        include: Any = None,
        skip: Optional[int] = None,
    ) -> Optional[SQLModel]:
        if include is not None:
            _warn_unsupported("include", self._model.__tablename__)
        clause = _translate_where(self._model, where)
        stmt = self._select_stmt()
        if clause is not None:
            stmt = stmt.where(clause)
        for clause_obj in _translate_order(self._model, order):
            stmt = stmt.order_by(clause_obj)
        if skip:
            stmt = stmt.offset(skip)
        stmt = stmt.limit(1)
        return await self._scalar_one_or_none(stmt)

    async def find_many(
        self,
        *,
        where: Optional[Mapping[str, Any]] = None,
        order: Any = None,
        take: Optional[int] = None,
        skip: Optional[int] = None,
        include: Any = None,
        select: Any = None,  # noqa: A002
    ) -> List[SQLModel]:
        if include is not None:
            _warn_unsupported("include", self._model.__tablename__)
        if select is not None:
            _warn_unsupported("select", self._model.__tablename__)
        clause = _translate_where(self._model, where)
        stmt = self._select_stmt()
        if clause is not None:
            stmt = stmt.where(clause)
        for clause_obj in _translate_order(self._model, order):
            stmt = stmt.order_by(clause_obj)
        if take:
            stmt = stmt.limit(take)
        if skip:
            stmt = stmt.offset(skip)
        return await self._scalars_all(stmt)

    async def count(
        self, *, where: Optional[Mapping[str, Any]] = None, **_: Any
    ) -> int:
        clause = _translate_where(self._model, where)
        stmt = select(func.count()).select_from(self._model)
        if clause is not None:
            stmt = stmt.where(clause)
        async with self._session() as sess:
            try:
                result = await sess.execute(stmt)
                return int(result.scalar_one())
            except SQLAlchemyError as exc:
                raise errors.map_sqlalchemy_error(exc) from exc

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def create(self, *, data: Mapping[str, Any], include: Any = None) -> SQLModel:
        if include is not None:
            _warn_unsupported("include", self._model.__tablename__)
        translated = _translate_data(self._model, data)
        instance = self._model(**translated)
        async with self._session() as sess:
            try:
                sess.add(instance)
                await sess.flush()
                await self._maybe_commit(sess)
            except SQLAlchemyError as exc:
                if not self._is_bound:
                    await sess.rollback()
                raise errors.map_sqlalchemy_error(exc) from exc
        return instance

    async def update(
        self,
        *,
        where: Mapping[str, Any],
        data: Mapping[str, Any],
        include: Any = None,
    ) -> Optional[SQLModel]:
        if include is not None:
            _warn_unsupported("include", self._model.__tablename__)
        clause = _translate_where(self._model, where)
        translated = _translate_data(self._model, data)
        async with self._session() as sess:
            try:
                stmt = self._select_stmt()
                if clause is not None:
                    stmt = stmt.where(clause)
                obj = (await sess.execute(stmt.limit(1))).scalar_one_or_none()
                if obj is None:
                    raise errors.RecordNotFoundError(
                        f"No {self._model.__name__} matched where={dict(where)}"
                    )
                for col, val in translated.items():
                    setattr(obj, col, val)
                await sess.flush()
                await self._maybe_commit(sess)
                return obj
            except SQLAlchemyError as exc:
                if not self._is_bound:
                    await sess.rollback()
                raise errors.map_sqlalchemy_error(exc) from exc

    async def upsert(
        self,
        *,
        where: Mapping[str, Any],
        data: Mapping[str, Any],
        include: Any = None,
    ) -> SQLModel:
        """Prisma's ``upsert`` takes ``data={"create": {...}, "update": {...}}``.

        We mirror that contract -- the existing call sites all use the
        nested ``create``/``update`` shape.
        """
        if include is not None:
            _warn_unsupported("include", self._model.__tablename__)
        if (
            not isinstance(data, Mapping)
            or "create" not in data
            or "update" not in data
        ):
            raise ValueError(
                "upsert(data=...) must be {'create': {...}, 'update': {...}}"
            )
        existing = await self.find_unique(where=where)
        if existing is None:
            create_payload = dict(data["create"])
            return await self.create(data=create_payload)
        return await self.update(where=where, data=data["update"])  # type: ignore[return-value]

    async def delete(self, *, where: Mapping[str, Any]) -> Optional[SQLModel]:
        clause = _translate_where(self._model, where)
        async with self._session() as sess:
            try:
                stmt = self._select_stmt()
                if clause is not None:
                    stmt = stmt.where(clause)
                obj = (await sess.execute(stmt.limit(1))).scalar_one_or_none()
                if obj is None:
                    raise errors.RecordNotFoundError(
                        f"No {self._model.__name__} matched where={dict(where)}"
                    )
                await sess.delete(obj)
                await self._maybe_commit(sess)
                return obj
            except SQLAlchemyError as exc:
                if not self._is_bound:
                    await sess.rollback()
                raise errors.map_sqlalchemy_error(exc) from exc

    async def update_many(
        self, *, where: Mapping[str, Any], data: Mapping[str, Any]
    ) -> CountResult:
        clause = _translate_where(self._model, where)
        translated = _translate_data(self._model, data)
        async with self._session() as sess:
            try:
                stmt = update(self._model).values(**translated)
                if clause is not None:
                    stmt = stmt.where(clause)
                result = await sess.execute(stmt)
                await self._maybe_commit(sess)
                return CountResult(count=result.rowcount or 0)
            except SQLAlchemyError as exc:
                if not self._is_bound:
                    await sess.rollback()
                raise errors.map_sqlalchemy_error(exc) from exc

    async def delete_many(
        self, *, where: Optional[Mapping[str, Any]] = None
    ) -> CountResult:
        clause = _translate_where(self._model, where)
        async with self._session() as sess:
            try:
                stmt = delete(self._model)
                if clause is not None:
                    stmt = stmt.where(clause)
                result = await sess.execute(stmt)
                await self._maybe_commit(sess)
                return CountResult(count=result.rowcount or 0)
            except SQLAlchemyError as exc:
                if not self._is_bound:
                    await sess.rollback()
                raise errors.map_sqlalchemy_error(exc) from exc

    async def create_many(
        self, *, data: Sequence[Mapping[str, Any]], skip_duplicates: bool = False
    ) -> CountResult:
        if not data:
            return CountResult(count=0)
        instances = [self._model(**_translate_data(self._model, item)) for item in data]
        async with self._session() as sess:
            try:
                sess.add_all(instances)
                await sess.flush()
                await self._maybe_commit(sess)
                return CountResult(count=len(instances))
            except SQLAlchemyError as exc:
                if not self._is_bound:
                    await sess.rollback()
                if skip_duplicates and isinstance(
                    errors.map_sqlalchemy_error(exc), errors.UniqueViolationError
                ):
                    return CountResult(count=0)
                raise errors.map_sqlalchemy_error(exc) from exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_stmt(self):
        return select(self._model)

    async def _scalar_one_or_none(self, stmt) -> Optional[SQLModel]:
        async with self._session() as sess:
            try:
                result = await sess.execute(stmt)
                return result.scalar_one_or_none()
            except SQLAlchemyError as exc:
                raise errors.map_sqlalchemy_error(exc) from exc

    async def _scalars_all(self, stmt) -> List[SQLModel]:
        async with self._session() as sess:
            try:
                result = await sess.execute(stmt)
                return list(result.scalars().all())
            except SQLAlchemyError as exc:
                raise errors.map_sqlalchemy_error(exc) from exc


# ---------------------------------------------------------------------------
# Batch / transaction
# ---------------------------------------------------------------------------


class _BatchOp:
    __slots__ = ("kind", "args", "kwargs", "model")

    def __init__(self, kind: str, model: Type[SQLModel], args, kwargs) -> None:
        self.kind = kind
        self.model = model
        self.args = args
        self.kwargs = kwargs


class _BatchTableAccessor:
    """Records ops queued against a particular table for later flush."""

    def __init__(self, model: Type[SQLModel], ops: List[_BatchOp]) -> None:
        self._model = model
        self._ops = ops

    def update(self, *args, **kwargs) -> None:
        self._ops.append(_BatchOp("update", self._model, args, kwargs))

    def upsert(self, *args, **kwargs) -> None:
        self._ops.append(_BatchOp("upsert", self._model, args, kwargs))

    def update_many(self, *args, **kwargs) -> None:
        self._ops.append(_BatchOp("update_many", self._model, args, kwargs))

    def create(self, *args, **kwargs) -> None:
        self._ops.append(_BatchOp("create", self._model, args, kwargs))

    def delete(self, *args, **kwargs) -> None:
        self._ops.append(_BatchOp("delete", self._model, args, kwargs))


class BatchAccessor:
    """Object returned by ``db.batch_()``.

    Mimics prisma-client-py's batcher: queue mutations against any table
    via ``batcher.litellm_<table>.<method>(...)``, then flush all of them
    in a single transaction with ``await batcher.commit()``.
    """

    def __init__(self, db: LiteLLMDB) -> None:
        self._db = db
        self._ops: List[_BatchOp] = []

    def __getattr__(self, name: str) -> _BatchTableAccessor:
        if name.startswith("_") or name in ("commit",):
            raise AttributeError(name)
        return _BatchTableAccessor(_model_for_accessor(name), self._ops)

    async def commit(self) -> None:
        if not self._ops:
            return
        async with self._db.session_ctx() as sess:
            try:
                async with sess.begin():
                    for op in self._ops:
                        await _execute_op_in_session(sess, op)
            except SQLAlchemyError as exc:
                raise errors.map_sqlalchemy_error(exc) from exc


class TransactionContext:
    """Object yielded by ``async with db.tx():``.

    Exposes table accessors that share a single session so that all the
    ops inside the ``with`` block execute in the same transaction.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def __getattr__(self, name: str) -> "TableAccessor":
        if name.startswith("_") or name in ("batch_",):
            raise AttributeError(name)
        model = _model_for_accessor(name)
        return TableAccessor(model, bound_session=self._session)

    def batch_(self) -> "TransactionBatch":
        return TransactionBatch(self._session)


class TransactionBatch:
    """``batch_`` inside an open transaction shares the same session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._ops: List[_BatchOp] = []

    def __getattr__(self, name: str) -> _BatchTableAccessor:
        if name.startswith("_") or name in ("commit",):
            raise AttributeError(name)
        return _BatchTableAccessor(_model_for_accessor(name), self._ops)

    async def commit(self) -> None:
        for op in self._ops:
            await _execute_op_in_session(self._session, op)
        # No commit() here -- the surrounding tx() context manager owns
        # transaction lifetime.


async def _execute_op_in_session(sess: AsyncSession, op: _BatchOp) -> None:
    """Execute a queued batch op against an open session.

    Mirrors :class:`TableAccessor` but never opens its own transaction.
    """
    model = op.model
    kwargs = op.kwargs
    if op.kind == "update":
        clause = _translate_where(model, kwargs.get("where"))
        translated = _translate_data(model, kwargs.get("data") or {})
        stmt = update(model).values(**translated)
        if clause is not None:
            stmt = stmt.where(clause)
        await sess.execute(stmt)
    elif op.kind == "update_many":
        clause = _translate_where(model, kwargs.get("where"))
        translated = _translate_data(model, kwargs.get("data") or {})
        stmt = update(model).values(**translated)
        if clause is not None:
            stmt = stmt.where(clause)
        await sess.execute(stmt)
    elif op.kind == "upsert":
        # Without ON CONFLICT awareness in the generic shim, fall back to
        # SELECT-then-update-or-insert. Sufficient for the call sites we
        # see, all of which use unique-keyed where clauses.
        clause = _translate_where(model, kwargs.get("where"))
        existing_stmt = select(model).limit(1)
        if clause is not None:
            existing_stmt = existing_stmt.where(clause)
        existing = (await sess.execute(existing_stmt)).scalar_one_or_none()
        data = kwargs.get("data") or {}
        if existing is None:
            create_payload = _translate_data(model, dict(data.get("create") or {}))
            sess.add(model(**create_payload))
        else:
            update_payload = _translate_data(model, dict(data.get("update") or {}))
            for col, val in update_payload.items():
                setattr(existing, col, val)
    elif op.kind == "create":
        translated = _translate_data(model, kwargs.get("data") or {})
        sess.add(model(**translated))
    elif op.kind == "delete":
        clause = _translate_where(model, kwargs.get("where"))
        stmt = delete(model)
        if clause is not None:
            stmt = stmt.where(clause)
        await sess.execute(stmt)
    else:
        raise NotImplementedError(f"Batch op kind '{op.kind}' is not implemented.")


# ---------------------------------------------------------------------------
# Top-level Prisma-compatible client
# ---------------------------------------------------------------------------


class PrismaCompatClient:
    """Replacement for prisma-client-py's ``Prisma`` class.

    Exposes a ``.db`` attribute (returns ``self`` -- the historical
    ``prisma_client.db.litellm_x`` shape collapses cleanly because the
    table accessors live on this object) plus connection lifecycle and
    raw-SQL methods.
    """

    def __init__(self, db: LiteLLMDB) -> None:
        self._db = db

    # The historical surface is ``prisma_client.db.litellm_x`` -- callers
    # do ``client.db.litellm_x`` *and* ``client.litellm_x`` interchangeably
    # in some places. We expose ``.db`` as ``self`` to match both.
    @property
    def db(self) -> "PrismaCompatClient":
        return self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await self._db.connect()

    async def disconnect(self) -> None:
        await self._db.disconnect()

    def is_connected(self) -> bool:
        return self._db.is_connected()

    async def start_token_refresh_task(self) -> None:
        await self._db.start_token_refresh_task()

    async def stop_token_refresh_task(self) -> None:
        await self._db.stop_token_refresh_task()

    # ------------------------------------------------------------------
    # Table accessor lookup
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> TableAccessor:
        # Accessor names are lowercase; the property ``db`` is intentionally
        # not routed here. Names starting with ``_`` are protected (so
        # SQLAlchemy / pydantic introspection doesn't trip the lookup).
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _TABLE_INDEX:
            return TableAccessor(_TABLE_INDEX[name], db=self._db)
        raise AttributeError(
            f"PrismaCompatClient has no attribute '{name}'. "
            f"Known accessors: {sorted(_TABLE_INDEX.keys())[:5]}..."
        )

    # ------------------------------------------------------------------
    # Raw SQL
    # ------------------------------------------------------------------

    async def query_raw(
        self, *args: Any, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        sql = query
        params: Iterable[Any] = ()
        if sql is None:
            if not args:
                raise ValueError("query_raw() requires either positional SQL or query=")
            sql = args[0]
            params = args[1:]
        else:
            params = args
        async with self._db.session_ctx() as sess:
            try:
                result = await sess.execute(_to_text(sql, params), _params_dict(params))
                rows = result.fetchall()
                return [_row_to_dict(r) for r in rows]
            except SQLAlchemyError as exc:
                raise errors.map_sqlalchemy_error(exc) from exc

    async def execute_raw(self, *args: Any, query: Optional[str] = None) -> int:
        sql = query
        params: Iterable[Any] = ()
        if sql is None:
            if not args:
                raise ValueError(
                    "execute_raw() requires either positional SQL or query="
                )
            sql = args[0]
            params = args[1:]
        else:
            params = args
        async with self._db.session_ctx() as sess:
            try:
                result = await sess.execute(_to_text(sql, params), _params_dict(params))
                await sess.commit()
                return result.rowcount or 0
            except SQLAlchemyError as exc:
                await sess.rollback()
                raise errors.map_sqlalchemy_error(exc) from exc

    # ------------------------------------------------------------------
    # Batch / tx
    # ------------------------------------------------------------------

    def batch_(self) -> BatchAccessor:
        return BatchAccessor(self._db)

    @asynccontextmanager
    async def tx(
        self, *, timeout: Optional[Any] = None, max_wait: Optional[Any] = None
    ) -> AsyncIterator[TransactionContext]:
        # ``timeout`` / ``max_wait`` are honoured by the underlying engine
        # statement_timeout (set on the connection) when configured; we do
        # not enforce them inside Python.
        del timeout, max_wait
        async with self._db.session_ctx() as sess:
            async with sess.begin():
                yield TransactionContext(sess)


# ---------------------------------------------------------------------------
# Raw-SQL parameter handling (Postgres ``$1``-style -> SQLAlchemy named)
# ---------------------------------------------------------------------------


def _to_text(sql: str, params: Iterable[Any]):
    """Rewrite ``$1``/``$2`` placeholders to SQLAlchemy ``:p1``/``:p2``.

    Prisma raw queries use Postgres-style positional placeholders. SQLAlchemy
    needs ``text(":p1")`` with bound params. We rewrite numerically-stable
    placeholders so the same SQL works on either backend.
    """
    if not params:
        return text(sql)
    rewritten = sql
    for i, _ in enumerate(params, start=1):
        rewritten = rewritten.replace(f"${i}", f":p{i}")
    return text(rewritten)


def _params_dict(params: Iterable[Any]) -> Dict[str, Any]:
    return {f"p{i}": v for i, v in enumerate(params, start=1)}


# ---------------------------------------------------------------------------
# Convenience: build the global client
# ---------------------------------------------------------------------------


def create_client(db: LiteLLMDB) -> PrismaCompatClient:
    """Factory used by ``litellm.proxy.utils.PrismaClient``."""
    return PrismaCompatClient(db)


# Silence unused-import warning for asyncio (used implicitly in cancellation
# semantics inside engine.py; keeping the import here documents that this
# module is async-aware end-to-end).
_ = asyncio
