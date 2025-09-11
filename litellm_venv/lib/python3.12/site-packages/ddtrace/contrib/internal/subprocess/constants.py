from ddtrace.appsec._constants import Constant_Class


class COMMANDS(metaclass=Constant_Class):
    """
    string names used by the library for tagging data for subprocess executions in context or span
    """

    SPAN_NAME = "command_execution"
    COMPONENT = "component"
    SHELL = "cmd.shell"
    EXEC = "cmd.exec"
    TRUNCATED = "cmd.truncated"
    EXIT_CODE = "cmd.exit_code"
    CTX_SUBP_IS_SHELL = "subprocess_popen_is_shell"
    CTX_SUBP_TRUNCATED = "subprocess_popen_truncated"
    CTX_SUBP_LINE = "subprocess_popen_line"
    CTX_SUBP_BINARY = "subprocess_popen_binary"
