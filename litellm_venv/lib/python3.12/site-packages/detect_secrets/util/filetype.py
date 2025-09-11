import os
from enum import Enum


class FileType(Enum):
    CLS = 0
    EXAMPLE = 1
    GO = 2
    JAVA = 3
    JAVASCRIPT = 4
    PHP = 5
    OBJECTIVE_C = 6
    PYTHON = 7
    SWIFT = 8
    TERRAFORM = 9
    YAML = 10
    C_SHARP = 11
    C = 12
    C_PLUS_PLUS = 13
    CONFIG = 14
    INI = 15
    PROPERTIES = 16
    TOML = 17
    OTHER = 18


def determine_file_type(filename: str) -> FileType:
    _, file_extension = os.path.splitext(filename)
    return {
        ".cls": FileType.CLS,
        ".example": FileType.EXAMPLE,
        ".eyaml": FileType.YAML,
        ".go": FileType.GO,
        ".java": FileType.JAVA,
        ".js": FileType.JAVASCRIPT,
        ".m": FileType.OBJECTIVE_C,
        ".php": FileType.PHP,
        ".py": FileType.PYTHON,
        ".pyi": FileType.PYTHON,
        ".swift": FileType.SWIFT,
        ".tf": FileType.TERRAFORM,
        ".yaml": FileType.YAML,
        ".yml": FileType.YAML,
        ".cs": FileType.C_SHARP,
        ".c": FileType.C,
        ".cpp": FileType.C_PLUS_PLUS,
        ".cnf": FileType.CONFIG,
        ".conf": FileType.CONFIG,
        ".cfg": FileType.CONFIG,
        ".cf": FileType.CONFIG,
        ".ini": FileType.INI,
        ".properties": FileType.PROPERTIES,
        ".toml": FileType.TOML,
    }.get(file_extension, FileType.OTHER)
