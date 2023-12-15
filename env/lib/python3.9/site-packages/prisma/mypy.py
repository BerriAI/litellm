import re
import copy
import logging
import builtins
import operator
from configparser import ConfigParser
from typing import (
    Optional,
    Callable,
    Dict,
    Any,
    Union,
    Type as TypingType,
    cast,
)

from mypy.options import Options
from mypy.errorcodes import ErrorCode
from mypy.types import (
    UnionType,
    NoneType,
    Type,
    Instance,
)
from mypy.nodes import (
    Node,
    Expression,
    DictExpr,
    StrExpr,
    NameExpr,
    Var,
    BytesExpr,
    CallExpr,
    IntExpr,
    Context,
    TypeInfo,
    SymbolTable,
    SymbolTableNode,
)
from mypy.plugin import Plugin, MethodContext, CheckerPluginInterface


# match any direct children of an actions class
CLIENT_ACTION_CHILD = re.compile(
    r'prisma\.actions\.(.*)Actions\.(?P<name>(((?!\.).)*$))'
)
ACTIONS = [
    'create',
    'find_unique',
    'delete',
    'update',
    'find_first',
    'find_many',
    'upsert',
    'update_many',
    'delete_many',
    'count',
]

CONFIGFILE_KEY = 'prisma-mypy'

log: logging.Logger = logging.getLogger(__name__)


# due to the way the mypy API is typed we unfortunately have to disable Pyright type checks
# this is because mypy type hints are written like this: `Bogus[str]` instead of `str`
# mypy uses internal magic to transform Bogus[T] to T which pyright cannot understand.
# pyright: reportGeneralTypeIssues=false, reportUnnecessaryComparison=false


def plugin(version: str) -> TypingType[Plugin]:
    return PrismaPlugin


class PrismaPluginConfig:
    __slots__ = ('warn_parsing_errors',)
    warn_parsing_errors: bool

    def __init__(self, options: Options) -> None:
        if options.config_file is None:  # pragma: no cover
            return

        plugin_config = ConfigParser()
        plugin_config.read(options.config_file)
        for key in self.__slots__:
            setting = plugin_config.getboolean(
                CONFIGFILE_KEY, key, fallback=True
            )
            setattr(self, key, setting)


class PrismaPlugin(Plugin):
    config: PrismaPluginConfig

    def __init__(self, options: Options) -> None:
        self.config = PrismaPluginConfig(options)
        super().__init__(options)

    def get_method_hook(
        self, fullname: str
    ) -> Optional[Callable[[MethodContext], Type]]:
        match = CLIENT_ACTION_CHILD.match(fullname)
        if not match:
            return None

        if match.group('name') in ACTIONS:
            return self.handle_action_invocation

        return None

    def handle_action_invocation(self, ctx: MethodContext) -> Type:
        # TODO: if an error occurs, log it so that we don't cause mypy to
        #       exit prematurely.
        return self._handle_include(ctx)

    def _handle_include(self, ctx: MethodContext) -> Type:
        """Recursively remove Optional from a relational field of a model
        if it was explicitly included.

        An argument could be made that this is over-engineered
        and while I do agree to an extent, the benefit of this
        method over just setting the default value to an empty list
        is that access to a relational field without explicitly
        including it will raise an error when type checking, e.g

        user = await client.user.find_unique(where={'id': user_id})
        print('\n'.join(p.title for p in user.posts))
        """
        include_expr = self.get_arg_named('include', ctx)
        if include_expr is None:
            return ctx.default_return_type

        if not isinstance(ctx.default_return_type, Instance):
            # TODO: resolve this?
            return ctx.default_return_type

        is_coroutine = self.is_coroutine_type(ctx.default_return_type)
        if is_coroutine:
            actual_ret = ctx.default_return_type.args[2]
        else:
            actual_ret = ctx.default_return_type

        is_optional = self.is_optional_type(actual_ret)
        if is_optional:
            actual_ret = cast(UnionType, actual_ret)
            model_type = actual_ret.items[0]
        else:
            model_type = actual_ret

        if not isinstance(model_type, Instance):
            return ctx.default_return_type

        try:
            include = self.parse_expression_to_dict(include_expr)
            new_model = self.modify_model_from_include(model_type, include)
        except Exception as exc:
            log.debug(
                'Ignoring %s exception while parsing include: %s',
                type(exc).__name__,
                exc,
            )

            # TODO: test this, pytest-mypy-plugins does not bode well with multiple line output
            if self.config.warn_parsing_errors:
                # TODO: add more details
                # e.g. "include" to "find_unique" of "UserActions"
                if isinstance(exc, UnparsedExpression):
                    err_ctx = exc.context
                else:
                    err_ctx = include_expr

                error_unable_to_parse(
                    ctx.api,
                    err_ctx,
                    'the "include" argument',
                )

            return ctx.default_return_type

        if is_optional:
            actual_ret = cast(UnionType, actual_ret)
            modified_ret = self.copy_modified_optional_type(
                actual_ret, new_model
            )
        else:
            modified_ret = new_model  # type: ignore

        if is_coroutine:
            arg1, arg2, _ = ctx.default_return_type.args
            return ctx.default_return_type.copy_modified(
                args=[arg1, arg2, modified_ret]
            )

        return modified_ret

    def modify_model_from_include(
        self, model: Instance, data: Dict[Any, Any]
    ) -> Instance:
        names = model.type.names.copy()
        for key, node in model.type.names.items():
            names[key] = self.maybe_modify_included_field(key, node, data)

        return self.copy_modified_instance(model, names)

    def maybe_modify_included_field(
        self,
        key: Union[str, Expression, Node],
        node: SymbolTableNode,
        data: Dict[Any, Any],
    ) -> SymbolTableNode:
        value = data.get(key)
        if value is False or value is None:
            return node

        if isinstance(value, (Expression, Node)):
            raise UnparsedExpression(value)

        # we do not want to remove the Optional from a field that is not a list
        # as the Optional indicates that the field is optional on a database level
        if (
            not isinstance(node.node, Var)
            or node.node.type is None
            or not isinstance(node.node.type, UnionType)
            or not self.is_optional_union_type(node.node.type)
            or not self.is_list_type(node.node.type.items[0])
        ):
            log.debug(
                'Not modifying included field: %s',
                key,
            )
            return node

        # this whole mess with copying is so that the modified field is not leaked
        new = node.copy()
        new.node = copy.copy(new.node)
        assert isinstance(new.node, Var)
        new.node.type = node.node.type.items[0]

        if (
            isinstance(value, dict)
            and 'include' in value
            and isinstance(new.node.type, Instance)
            and isinstance(new.node.type.args[0], Instance)
        ):
            model = self.modify_model_from_include(
                new.node.type.args[0], value['include']
            )
            new.node.type.args = (model, *new.node.type.args)

        return new

    def get_arg_named(
        self, name: str, ctx: MethodContext
    ) -> Optional[Expression]:
        """Return the expression for an argument."""
        # keyword arguments
        for i, names in enumerate(ctx.arg_names):
            for j, arg_name in enumerate(names):
                if arg_name == name:
                    return ctx.args[i][j]

        # positional arguments
        for i, arg_name in enumerate(ctx.callee_arg_names):
            if arg_name == name and ctx.args[i]:
                return ctx.args[i][0]

        return None

    def is_optional_type(self, typ: Type) -> bool:
        return isinstance(typ, UnionType) and self.is_optional_union_type(typ)

    def is_optional_union_type(self, typ: UnionType) -> bool:
        return len(typ.items) == 2 and isinstance(typ.items[1], NoneType)

    # TODO: why is fullname Any?

    def is_coroutine_type(self, typ: Instance) -> bool:
        return bool(typ.type.fullname == 'typing.Coroutine')

    def is_list_type(self, typ: Type) -> bool:
        return (
            isinstance(typ, Instance) and typ.type.fullname == 'builtins.list'
        )

    def is_dict_call_type(self, expr: NameExpr) -> bool:
        # statically wise, TypedDicts do not inherit from dict
        # so we cannot check that, just checking if the expression
        # inherits from a class that ends with dict is good enough
        # for our use case
        return bool(expr.fullname == 'builtins.dict') or bool(
            isinstance(expr.node, TypeInfo)
            and expr.node.bases
            and expr.node.bases[0].type.fullname.lower().endswith('dict')
        )

    def copy_modified_instance(
        self, instance: Instance, names: SymbolTable
    ) -> Instance:
        new = copy.copy(instance)
        new.type = TypeInfo(names, new.type.defn, new.type.module_name)
        new.type.mro = [new.type, *instance.type.mro]
        new.type.bases = instance.type.bases
        new.type.metaclass_type = instance.type.metaclass_type
        return new

    def copy_modified_optional_type(
        self, original: UnionType, typ: Type
    ) -> UnionType:
        new = copy.copy(original)
        new.items = new.items.copy()
        new.items[0] = typ
        return new

    def parse_expression_to_dict(
        self, expression: Expression
    ) -> Dict[Any, Any]:
        if isinstance(expression, DictExpr):
            return self._dictexpr_to_dict(expression)

        if isinstance(expression, CallExpr):
            return self._callexpr_to_dict(expression)

        raise TypeError(
            f'Cannot parse expression of type={type(expression).__name__} to a dictionary.'
        )

    def _dictexpr_to_dict(self, expr: DictExpr) -> Dict[Any, Any]:
        parsed = {}
        for key_expr, value_expr in expr.items:
            if key_expr is None:
                # TODO: what causes this?
                continue

            key = self._resolve_expression(key_expr)
            value = self._resolve_expression(value_expr)
            parsed[key] = value

        return parsed

    def _callexpr_to_dict(
        self, expr: CallExpr, strict: bool = True
    ) -> Dict[str, Any]:
        if not isinstance(expr.callee, NameExpr):
            raise TypeError(
                f'Expected CallExpr.callee to be a NameExpr but got {type(expr.callee)} instead.'
            )

        if strict and not self.is_dict_call_type(expr.callee):
            raise TypeError(
                f'Expected builtins.dict to be called but got {expr.callee.fullname} instead'
            )

        parsed = {}
        for arg_name, value_expr in zip(expr.arg_names, expr.args):
            if arg_name is None:
                continue

            value = self._resolve_expression(value_expr)
            parsed[arg_name] = value

        return parsed

    def _resolve_expression(self, expression: Expression) -> Any:
        if isinstance(expression, (StrExpr, BytesExpr, IntExpr)):
            return expression.value

        if isinstance(expression, NameExpr):
            return self._resolve_name_expression(expression)

        if isinstance(expression, DictExpr):
            return self._dictexpr_to_dict(expression)

        if isinstance(expression, CallExpr):
            return self._callexpr_to_dict(expression)

        return expression

    def _resolve_name_expression(self, expression: NameExpr) -> Any:
        if isinstance(expression.node, Var):
            return self._resolve_var_node(expression.node)

        return expression

    def _resolve_var_node(self, node: Var) -> Any:
        if node.is_final:
            return node.final_value

        if node.fullname.startswith('builtins.'):
            return self._resolve_builtin(node.fullname)

        return node

    def _resolve_builtin(self, fullname: str) -> Any:
        return operator.attrgetter(*fullname.split('.')[1:])(builtins)


class UnparsedExpression(Exception):
    context: Union[Expression, Node]

    def __init__(self, context: Union[Expression, Node]) -> None:
        self.context = context
        super().__init__(
            f'Tried to access a ({type(context).__name__}) expression that was not parsed.'
        )


ERROR_PARSING = ErrorCode('prisma-parsing', 'Unable to parse', 'Prisma')


def error_unable_to_parse(
    api: CheckerPluginInterface, context: Context, detail: str
) -> None:
    link = (
        'https://github.com/RobertCraigie/prisma-client-py/issues/new/choose'
    )
    full_message = f'The prisma mypy plugin was unable to parse: {detail}\n'
    full_message += f'Please consider reporting this bug at {link} so we can try to fix it!'
    api.fail(full_message, context, code=ERROR_PARSING)
