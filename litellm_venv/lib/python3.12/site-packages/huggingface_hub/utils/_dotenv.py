# AI-generated module (ChatGPT)
import re
from typing import Dict, Optional


def load_dotenv(
    dotenv_str: str, environ: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Parse a DOTENV-format string and return a dictionary of key-value pairs.
    Handles quoted values, comments, export keyword, and blank lines.
    """
    env: Dict[str, str] = {}
    line_pattern = re.compile(
        r"""
        ^\s*
        (?:export[^\S\n]+)?               # optional export
        ([A-Za-z_][A-Za-z0-9_]*)          # key
        [^\S\n]*(=)?[^\S\n]*
        (                                 # value group
            (?:
                '(?:\\'|[^'])*'           # single-quoted value
                | \"(?:\\\"|[^\"])*\"     # double-quoted value
                | [^#\n\r]+?              # unquoted value
            )
        )?
        [^\S\n]*(?:\#.*)?$                # optional inline comment
    """,
        re.VERBOSE,
    )

    for line in dotenv_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # Skip comments and empty lines

        match = line_pattern.match(line)
        if match:
            key = match.group(1)
            val = None
            if match.group(2):  # if there is '='
                raw_val = match.group(3) or ""
                val = raw_val.strip()
                # Remove surrounding quotes if quoted
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                    val = (
                        val.replace(r"\n", "\n")
                        .replace(r"\t", "\t")
                        .replace(r"\"", '"')
                        .replace(r"\\", "\\")
                    )
                    if raw_val.startswith('"'):
                        val = val.replace(r"\$", "$")  # only in double quotes
            elif environ is not None:
                # Get it from the current environment
                val = environ.get(key)

            if val is not None:
                env[key] = val

    return env
