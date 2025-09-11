from collections import deque
from collections import namedtuple
import os
import typing as t
import warnings


class NoDefaultType(object):
    def __str__(self):
        return ""


NoDefault = NoDefaultType()
DeprecationInfo = t.Tuple[str, str, str]


T = t.TypeVar("T")
K = t.TypeVar("K")
V = t.TypeVar("V")

MapType = t.Union[t.Callable[[str], V], t.Callable[[str, str], t.Tuple[K, V]]]
HelpInfo = namedtuple("HelpInfo", ("name", "type", "default", "help"))


def _normalized(name: str) -> str:
    return name.upper().replace(".", "_").rstrip("_")


def _check_type(value: t.Any, _type: t.Union[object, t.Type[T]]) -> bool:
    if hasattr(_type, "__origin__"):
        return isinstance(value, _type.__args__)  # type: ignore[attr-defined,union-attr]

    return isinstance(value, _type)  # type: ignore[arg-type]


class EnvVariable(t.Generic[T]):
    def __init__(
        self,
        type: t.Union[object, t.Type[T]],
        name: str,
        parser: t.Optional[t.Callable[[str], T]] = None,
        validator: t.Optional[t.Callable[[T], None]] = None,
        map: t.Optional[MapType] = None,
        default: t.Union[T, NoDefaultType] = NoDefault,
        deprecations: t.Optional[t.List[DeprecationInfo]] = None,
        private: bool = False,
        help: t.Optional[str] = None,
        help_type: t.Optional[str] = None,
        help_default: t.Optional[str] = None,
    ) -> None:
        if hasattr(type, "__origin__") and type.__origin__ is t.Union:  # type: ignore[attr-defined,union-attr]
            if not isinstance(default, type.__args__):  # type: ignore[attr-defined,union-attr]
                raise TypeError(
                    "default must be either of these types {}".format(type.__args__)  # type: ignore[attr-defined,union-attr]
                )
        elif default is not NoDefault and not isinstance(default, type):  # type: ignore[arg-type]
            raise TypeError("default must be of type {}".format(type))

        self.type = type
        self.name = name
        self.parser = parser
        self.validator = validator
        self.map = map
        self.default = default
        self.deprecations = deprecations
        self.private = private

        self.help = help
        self.help_type = help_type
        self.help_default = help_default

        self._full_name = _normalized(name)  # Will be set by the EnvMeta metaclass

    @property
    def full_name(self) -> str:
        return f"_{self._full_name}" if self.private else self._full_name

    def _cast(self, _type: t.Any, raw: str, env: "Env") -> t.Any:
        if _type is bool:
            return t.cast(T, raw.lower() in env.__truthy__)
        elif _type in (list, tuple, set):
            collection = raw.split(env.__item_separator__)
            return t.cast(
                T,
                _type(  # type: ignore[operator]
                    collection if self.map is None else map(self.map, collection)  # type: ignore[arg-type]
                ),
            )
        elif _type is dict:
            d = dict(
                _.split(env.__value_separator__, 1)
                for _ in raw.split(env.__item_separator__)
            )
            if self.map is not None:
                d = dict(self.map(*_) for _ in d.items())
            return t.cast(T, d)

        if _check_type(raw, _type):
            return t.cast(T, raw)

        try:
            return _type(raw)
        except Exception as e:
            msg = f"cannot cast {raw} to {self.type}"
            raise TypeError(msg) from e

    def _retrieve(self, env: "Env", prefix: str) -> T:
        source = env.source

        full_name = self.full_name
        raw = source.get(full_name.format(**env.dynamic))
        if raw is None and self.deprecations:
            for name, deprecated_when, removed_when in self.deprecations:
                full_deprecated_name = prefix + _normalized(name)
                if self.private:
                    full_deprecated_name = f"_{full_deprecated_name}"
                raw = source.get(full_deprecated_name.format(**env.dynamic))
                if raw is not None:
                    deprecated_when_message = (
                        " in version %s" % deprecated_when
                        if deprecated_when is not None
                        else ""
                    )
                    removed_when_message = (
                        " and will be removed in version %s" % removed_when
                        if removed_when is not None
                        else ""
                    )
                    warnings.warn(
                        "%s has been deprecated%s%s. Use %s instead"
                        % (
                            full_deprecated_name,
                            deprecated_when_message,
                            removed_when_message,
                            full_name,
                        ),
                        DeprecationWarning,
                    )
                    break

        if raw is None:
            if not isinstance(self.default, NoDefaultType):
                return self.default

            raise KeyError(
                "Mandatory environment variable {} is not set".format(full_name)
            )

        if self.parser is not None:
            parsed = self.parser(raw)
            if not _check_type(parsed, self.type):
                raise TypeError(
                    "parser returned type {} instead of {}".format(
                        type(parsed), self.type
                    )
                )
            return parsed

        if hasattr(self.type, "__origin__") and self.type.__origin__ is t.Union:  # type: ignore[attr-defined,union-attr]
            for ot in self.type.__args__:  # type: ignore[attr-defined,union-attr]
                try:
                    return t.cast(T, self._cast(ot, raw, env))
                except TypeError:
                    pass

        return self._cast(self.type, raw, env)

    def __call__(self, env: "Env", prefix: str) -> T:
        value = self._retrieve(env, prefix)

        if self.validator is not None:
            try:
                self.validator(value)
            except ValueError as e:
                msg = f"Invalid value for environment variable {self.full_name}: {e}"
                raise ValueError(msg)

        return value


class DerivedVariable(t.Generic[T]):
    def __init__(self, type: t.Type[T], derivation: t.Callable[["Env"], T]) -> None:
        self.type = type
        self.derivation = derivation

    def __call__(self, env: "Env") -> T:
        value = self.derivation(env)
        if not _check_type(value, self.type):
            raise TypeError(
                "derivation returned type {} instead of {}".format(
                    type(value), self.type
                )
            )
        return value


class EnvMeta(type):
    def __new__(
        cls, name: str, bases: t.Tuple[t.Type], ns: t.Dict[str, t.Any]
    ) -> t.Any:
        env = t.cast("Env", super().__new__(cls, name, bases, ns))

        prefix = ns.get("__prefix__")
        if prefix:
            for v in env.values(recursive=True):
                if isinstance(v, EnvVariable):
                    v._full_name = f"{_normalized(prefix)}_{v._full_name}".upper()

        return env


class Env(metaclass=EnvMeta):
    """Env base class.

    This class is meant to be subclassed. The configuration is declared by using
    the ``Env.var`` and ``Env.der`` class methods. The former declares a mapping
    between attributes of the instance of the subclass with the environment
    variables. The latter declares derived attributes that are computed using
    a given derivation function.

    If variables share a common prefix, this can be specified with the
    ``__prefix__`` class attribute. t.Any dots in the prefix or the variable names
    will be replaced with underscores. The variable names will be uppercased
    before being looked up in the environment.

    By default, boolean variables evaluate to true if their lower-case value is
    one of ``true``, ``yes``, ``on`` or ``1``. This can be overridden by either
    passing a custom parser to the variable declaration, or by overriding the
    ``__truthy__`` class attribute, which is a set of lower-case strings that
    are considered to be a representation of ``True``.

    There is also basic support for collections. An item of type ``list``,
    ``t.Tuple`` or ``set`` will be parsed using ``,`` as item separator.
    Similarly, an item of type ``dict`` will be parsed with ``,`` as item
    separator, and ``:`` as value separator. These can be changed by overriding
    the ``__item_separator__`` and ``__value_separator__`` class attributes
    respectively. All the elements in the collections, including key and values
    for dictionaries, will be of type string. For more advanced control over
    the final type, a custom ``parser`` can be passed instead.
    """

    __truthy__ = frozenset({"1", "true", "yes", "on"})
    __prefix__ = ""
    __item__: t.Optional[str] = None
    __item_separator__ = ","
    __value_separator__ = ":"

    def __init__(
        self,
        source: t.Optional[t.Dict[str, str]] = None,
        parent: t.Optional["Env"] = None,
        dynamic: t.Optional[t.Dict[str, str]] = None,
    ) -> None:
        self.source = source or os.environ
        self.parent = parent
        self.dynamic = (
            {k.upper(): v.upper() for k, v in dynamic.items()}
            if dynamic is not None
            else {}
        )

        self._full_prefix: str = (
            parent._full_prefix if parent is not None else ""
        ) + _normalized(self.__prefix__)
        if self._full_prefix and not self._full_prefix.endswith("_"):
            self._full_prefix += "_"

        self.spec = self.__class__
        derived = []
        for name, e in list(self.__class__.__dict__.items()):
            if isinstance(e, EnvVariable):
                setattr(self, name, e(self, self._full_prefix))
            elif isinstance(e, type) and issubclass(e, Env):
                if e.__item__ is not None and e.__item__ != name:
                    # Move the subclass to the __item__ attribute
                    setattr(self.spec, e.__item__, e)
                    delattr(self.spec, name)
                    name = e.__item__
                setattr(self, name, e(source, self))
            elif isinstance(e, DerivedVariable):
                derived.append((name, e))

        for n, d in derived:
            setattr(self, n, d(self))

    @classmethod
    def var(
        cls,
        type: t.Type[T],
        name: str,
        parser: t.Optional[t.Callable[[str], T]] = None,
        validator: t.Optional[t.Callable[[T], None]] = None,
        map: t.Optional[MapType] = None,
        default: t.Union[T, NoDefaultType] = NoDefault,
        deprecations: t.Optional[t.List[DeprecationInfo]] = None,
        private: bool = False,
        help: t.Optional[str] = None,
        help_type: t.Optional[str] = None,
        help_default: t.Optional[str] = None,
    ) -> EnvVariable[T]:
        return EnvVariable(
            type,
            name,
            parser,
            validator,
            map,
            default,
            deprecations,
            private,
            help,
            help_type,
            help_default,
        )

    @classmethod
    def v(
        cls,
        type: t.Union[object, t.Type[T]],
        name: str,
        parser: t.Optional[t.Callable[[str], T]] = None,
        validator: t.Optional[t.Callable[[T], None]] = None,
        map: t.Optional[MapType] = None,
        default: t.Union[T, NoDefaultType] = NoDefault,
        deprecations: t.Optional[t.List[DeprecationInfo]] = None,
        private: bool = False,
        help: t.Optional[str] = None,
        help_type: t.Optional[str] = None,
        help_default: t.Optional[str] = None,
    ) -> EnvVariable[T]:
        return EnvVariable(
            type,
            name,
            parser,
            validator,
            map,
            default,
            deprecations,
            private,
            help,
            help_type,
            help_default,
        )

    @classmethod
    def der(
        cls, type: t.Type[T], derivation: t.Callable[["Env"], T]
    ) -> DerivedVariable[T]:
        return DerivedVariable(type, derivation)

    @classmethod
    def d(
        cls, type: t.Type[T], derivation: t.Callable[["Env"], T]
    ) -> DerivedVariable[T]:
        return DerivedVariable(type, derivation)

    @classmethod
    def items(
        cls, recursive: bool = False, include_derived: bool = False
    ) -> t.Iterator[t.Tuple[str, t.Union[EnvVariable, DerivedVariable]]]:
        classes = (EnvVariable, DerivedVariable) if include_derived else (EnvVariable,)
        q: t.Deque[t.Tuple[t.Tuple[str], t.Type["Env"]]] = deque()
        path: t.Tuple[str] = tuple()  # type: ignore[assignment]
        q.append((path, cls))
        while q:
            path, env = q.popleft()
            for k, v in env.__dict__.items():
                if isinstance(v, classes):
                    yield (
                        ".".join((*path, k)),
                        t.cast(t.Union[EnvVariable, DerivedVariable], v),
                    )
                elif isinstance(v, type) and issubclass(v, Env) and recursive:
                    item_name = getattr(v, "__item__", k)
                    if item_name is None:
                        item_name = k
                    q.append(((*path, item_name), v))  # type: ignore[arg-type]

    @classmethod
    def keys(
        cls, recursive: bool = False, include_derived: bool = False
    ) -> t.Iterator[str]:
        """Return the name of all the configuration items."""
        for k, _ in cls.items(recursive, include_derived):
            yield k

    @classmethod
    def values(
        cls, recursive: bool = False, include_derived: bool = False
    ) -> t.Iterator[t.Union[EnvVariable, DerivedVariable, t.Type["Env"]]]:
        """Return the value of all the configuration items."""
        for _, v in cls.items(recursive, include_derived):
            yield v

    @classmethod
    def include(
        cls,
        env_spec: t.Type["Env"],
        namespace: t.Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Include variables from another Env subclass.

        The new items can be merged at the top level, or parented to a
        namespace. By default, the method raises a ``ValueError`` if the
        operation would result in some variables being overwritten. This can
        be disabled by setting the ``overwrite`` argument to ``True``.
        """
        # Pick only the attributes that define variables.
        to_include = {
            k: v
            for k, v in env_spec.__dict__.items()
            if isinstance(v, (EnvVariable, DerivedVariable))
            or isinstance(v, type)
            and issubclass(v, Env)
        }

        own_prefix = _normalized(getattr(cls, "__prefix__", ""))

        if namespace is not None:
            if not overwrite and hasattr(cls, namespace):
                raise ValueError("Namespace already in use: {}".format(namespace))

            if getattr(cls, namespace, None) is not env_spec:
                setattr(cls, namespace, env_spec)

                if own_prefix:
                    for _, v in to_include.items():
                        if isinstance(v, EnvVariable):
                            v._full_name = f"{own_prefix}_{v._full_name}"

            return None

        if not overwrite:
            overlap = set(cls.__dict__.keys()) & set(to_include.keys())
            if overlap:
                raise ValueError("Configuration clashes detected: {}".format(overlap))

        other_prefix = getattr(env_spec, "__prefix__", "")
        for k, v in to_include.items():
            if getattr(cls, k, None) is not v:
                setattr(cls, k, v)
                if isinstance(v, EnvVariable):
                    if other_prefix:
                        v._full_name = v._full_name[len(other_prefix) + 1 :]  # noqa
                    if own_prefix:
                        v._full_name = f"{own_prefix}_{v._full_name}"

    @classmethod
    def help_info(
        cls, recursive: bool = False, include_private: bool = False
    ) -> t.List[HelpInfo]:
        """Extract the help information from the class.

        Returns a list of all the environment variables declared by the class.
        The format of each entry is a t.Tuple consisting of the variable name (in
        double backtics quotes), the type, the default value, and the help text.

        Set ``recursive`` to ``True`` to include variables from nested Env
        classes.

        Set ``include_private`` to ``True`` to include variables that are
        marked as private (i.e. their name starts with an underscore).
        """
        entries = []

        def add_entries(full_prefix: str, config: t.Type[Env]) -> None:
            vars = sorted(
                (_ for _ in config.values() if isinstance(_, EnvVariable)),
                key=lambda v: v.name,
            )

            for v in vars:
                if not include_private and v.private:
                    continue

                # Add a period at the end if necessary.
                help_message = v.help.strip() if v.help is not None else ""
                if help_message and not help_message.endswith("."):
                    help_message += "."

                if v.help_type is not None:
                    help_type = v.help_type
                else:
                    try:
                        help_type = v.type.__name__  # type: ignore[attr-defined]
                    except AttributeError:
                        # typing.t.Union[<type>, NoneType]
                        help_type = v.type.__args__[0].__name__  # type: ignore[attr-defined]

                private_prefix = "_" if v.private else ""

                entries.append(
                    HelpInfo(
                        f"{private_prefix}{full_prefix}{_normalized(v.name)}",
                        help_type,  # type: ignore[attr-defined]
                        (
                            v.help_default
                            if v.help_default is not None
                            else str(v.default)
                        ),
                        help_message,
                    )
                )

        configs = [("", cls)]

        while configs:
            full_prefix, config = configs.pop()
            new_prefix = full_prefix + _normalized(config.__prefix__)
            if new_prefix and not new_prefix.endswith("_"):
                new_prefix += "_"
            add_entries(new_prefix, config)

            if not recursive:
                break

            subconfigs = sorted(
                (
                    (new_prefix, v)
                    for k, v in config.__dict__.items()
                    if isinstance(v, type) and issubclass(v, Env) and k != "parent"
                ),
                key=lambda _: _[1].__prefix__,
            )

            configs[0:0] = subconfigs  # DFS

        return entries
