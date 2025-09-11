# coding: utf-8
"""
Sphinx directive integration
============================

We usually need to document the life-cycle of functions and classes:
when they are created, modified or deprecated.

To do that, `Sphinx <http://www.sphinx-doc.org>`_ has a set
of `Paragraph-level markups <http://www.sphinx-doc.org/en/stable/markup/para.html>`_:

- ``versionadded``: to document the version of the project which added the described feature to the library,
- ``versionchanged``: to document changes of a feature,
- ``deprecated``: to document a deprecated feature.

The purpose of this module is to defined decorators which adds this Sphinx directives
to the docstring of your function and classes.

Of course, the ``@deprecated`` decorator will emit a deprecation warning
when the function/method is called or the class is constructed.
"""
import re
import textwrap

from deprecated.classic import ClassicAdapter
from deprecated.classic import deprecated as _classic_deprecated


class SphinxAdapter(ClassicAdapter):
    """
    Sphinx adapter -- *for advanced usage only*

    This adapter override the :class:`~deprecated.classic.ClassicAdapter`
    in order to add the Sphinx directives to the end of the function/class docstring.
    Such a directive is a `Paragraph-level markup <http://www.sphinx-doc.org/en/stable/markup/para.html>`_

    - The directive can be one of "versionadded", "versionchanged" or "deprecated".
    - The version number is added if provided.
    - The reason message is obviously added in the directive block if not empty.
    """

    def __init__(
        self,
        directive,
        reason="",
        version="",
        action=None,
        category=DeprecationWarning,
        extra_stacklevel=0,
        line_length=70,
    ):
        """
        Construct a wrapper adapter.

        :type  directive: str
        :param directive:
            Sphinx directive: can be one of "versionadded", "versionchanged" or "deprecated".

        :type  reason: str
        :param reason:
            Reason message which documents the deprecation in your library (can be omitted).

        :type  version: str
        :param version:
            Version of your project which deprecates this feature.
            If you follow the `Semantic Versioning <https://semver.org/>`_,
            the version number has the format "MAJOR.MINOR.PATCH".

        :type  action: Literal["default", "error", "ignore", "always", "module", "once"]
        :param action:
            A warning filter used to activate or not the deprecation warning.
            Can be one of "error", "ignore", "always", "default", "module", or "once".
            If ``None`` or empty, the global filtering mechanism is used.
            See: `The Warnings Filter`_ in the Python documentation.

        :type  category: Type[Warning]
        :param category:
            The warning category to use for the deprecation warning.
            By default, the category class is :class:`~DeprecationWarning`,
            you can inherit this class to define your own deprecation warning category.

        :type  extra_stacklevel: int
        :param extra_stacklevel:
            Number of additional stack levels to consider instrumentation rather than user code.
            With the default value of 0, the warning refers to where the class was instantiated
            or the function was called.

        :type  line_length: int
        :param line_length:
            Max line length of the directive text. If non nul, a long text is wrapped in several lines.

        .. versionchanged:: 1.2.15
            Add the *extra_stacklevel* parameter.
        """
        if not version:
            # https://github.com/laurent-laporte-pro/deprecated/issues/40
            raise ValueError("'version' argument is required in Sphinx directives")
        self.directive = directive
        self.line_length = line_length
        super(SphinxAdapter, self).__init__(
            reason=reason,
            version=version,
            action=action,
            category=category,
            extra_stacklevel=extra_stacklevel,
        )

    def __call__(self, wrapped):
        """
        Add the Sphinx directive to your class or function.

        :param wrapped: Wrapped class or function.

        :return: the decorated class or function.
        """
        # -- build the directive division
        fmt = ".. {directive}:: {version}" if self.version else ".. {directive}::"
        div_lines = [fmt.format(directive=self.directive, version=self.version)]
        width = self.line_length - 3 if self.line_length > 3 else 2**16
        reason = textwrap.dedent(self.reason).strip()
        for paragraph in reason.splitlines():
            if paragraph:
                div_lines.extend(
                    textwrap.fill(
                        paragraph,
                        width=width,
                        initial_indent="   ",
                        subsequent_indent="   ",
                    ).splitlines()
                )
            else:
                div_lines.append("")

        # -- get the docstring, normalize the trailing newlines
        # keep a consistent behaviour if the docstring starts with newline or directly on the first one
        docstring = wrapped.__doc__ or ""
        lines = docstring.splitlines(True) or [""]
        docstring = textwrap.dedent("".join(lines[1:])) if len(lines) > 1 else ""
        docstring = lines[0] + docstring
        if docstring:
            # An empty line must separate the original docstring and the directive.
            docstring = re.sub(r"\n+$", "", docstring, flags=re.DOTALL) + "\n\n"
        else:
            # Avoid "Explicit markup ends without a blank line" when the decorated function has no docstring
            docstring = "\n"

        # -- append the directive division to the docstring
        docstring += "".join("{}\n".format(line) for line in div_lines)

        wrapped.__doc__ = docstring
        if self.directive in {"versionadded", "versionchanged"}:
            return wrapped
        return super(SphinxAdapter, self).__call__(wrapped)

    def get_deprecated_msg(self, wrapped, instance):
        """
        Get the deprecation warning message (without Sphinx cross-referencing syntax) for the user.

        :param wrapped: Wrapped class or function.

        :param instance: The object to which the wrapped function was bound when it was called.

        :return: The warning message.

        .. versionadded:: 1.2.12
           Strip Sphinx cross-referencing syntax from warning message.

        """
        msg = super(SphinxAdapter, self).get_deprecated_msg(wrapped, instance)
        # Strip Sphinx cross-reference syntax (like ":function:", ":py:func:" and ":py:meth:")
        # Possible values are ":role:`foo`", ":domain:role:`foo`"
        # where ``role`` and ``domain`` should match "[a-zA-Z]+"
        msg = re.sub(
            r"(?: : [a-zA-Z]+ )? : [a-zA-Z]+ : (`[^`]*`)", r"\1", msg, flags=re.X
        )
        return msg


def versionadded(reason="", version="", line_length=70):
    """
    This decorator can be used to insert a "versionadded" directive
    in your function/class docstring in order to document the
    version of the project which adds this new functionality in your library.

    :param str reason:
        Reason message which documents the addition in your library (can be omitted).

    :param str version:
        Version of your project which adds this feature.
        If you follow the `Semantic Versioning <https://semver.org/>`_,
        the version number has the format "MAJOR.MINOR.PATCH", and,
        in the case of a new functionality, the "PATCH" component should be "0".

    :type  line_length: int
    :param line_length:
        Max line length of the directive text. If non nul, a long text is wrapped in several lines.

    :return: the decorated function.
    """
    adapter = SphinxAdapter(
        "versionadded",
        reason=reason,
        version=version,
        line_length=line_length,
    )
    return adapter


def versionchanged(reason="", version="", line_length=70):
    """
    This decorator can be used to insert a "versionchanged" directive
    in your function/class docstring in order to document the
    version of the project which modifies this functionality in your library.

    :param str reason:
        Reason message which documents the modification in your library (can be omitted).

    :param str version:
        Version of your project which modifies this feature.
        If you follow the `Semantic Versioning <https://semver.org/>`_,
        the version number has the format "MAJOR.MINOR.PATCH".

    :type  line_length: int
    :param line_length:
        Max line length of the directive text. If non nul, a long text is wrapped in several lines.

    :return: the decorated function.
    """
    adapter = SphinxAdapter(
        "versionchanged",
        reason=reason,
        version=version,
        line_length=line_length,
    )
    return adapter


def deprecated(reason="", version="", line_length=70, **kwargs):
    """
    This decorator can be used to insert a "deprecated" directive
    in your function/class docstring in order to document the
    version of the project which deprecates this functionality in your library.

    :param str reason:
        Reason message which documents the deprecation in your library (can be omitted).

    :param str version:
        Version of your project which deprecates this feature.
        If you follow the `Semantic Versioning <https://semver.org/>`_,
        the version number has the format "MAJOR.MINOR.PATCH".

    :type  line_length: int
    :param line_length:
        Max line length of the directive text. If non nul, a long text is wrapped in several lines.

    Keyword arguments can be:

    -   "action":
        A warning filter used to activate or not the deprecation warning.
        Can be one of "error", "ignore", "always", "default", "module", or "once".
        If ``None``, empty or missing, the global filtering mechanism is used.

    -   "category":
        The warning category to use for the deprecation warning.
        By default, the category class is :class:`~DeprecationWarning`,
        you can inherit this class to define your own deprecation warning category.

    -   "extra_stacklevel":
        Number of additional stack levels to consider instrumentation rather than user code.
        With the default value of 0, the warning refers to where the class was instantiated
        or the function was called.


    :return: a decorator used to deprecate a function.

    .. versionchanged:: 1.2.13
       Change the signature of the decorator to reflect the valid use cases.

    .. versionchanged:: 1.2.15
        Add the *extra_stacklevel* parameter.
    """
    directive = kwargs.pop("directive", "deprecated")
    adapter_cls = kwargs.pop("adapter_cls", SphinxAdapter)
    kwargs["reason"] = reason
    kwargs["version"] = version
    kwargs["line_length"] = line_length
    return _classic_deprecated(directive=directive, adapter_cls=adapter_cls, **kwargs)
