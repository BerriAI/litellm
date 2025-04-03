import sys
import ast
import os

# Extremely restrictive set of allowed numbers
ALLOWED_NUMBERS = {
    0,
    1,
    -1,
    2,
    10,
    100,
    1000,
    4,
    3,
    500,
    6,
    60,
    3600,
    0.75,
    7,
    1024,
    1011,
    600,
    12,
    1000000000.0,
    0.1,
    50,
    128,
    6000,
    30,
    1000000,
    5,
    15,
    25,
    10000,
    60000,
    8,
    2048,
    16000000000,
    16,
    16383,
    14,
    24,
    128000,
    0.01,
    20,
}

# Add all standard HTTP status codes
HTTP_STATUS_CODES = {
    200,  # OK
    201,  # Created
    202,  # Accepted
    204,  # No Content
    300,  # Multiple Choices
    301,  # Moved Permanently
    302,  # Found
    303,  # See Other
    304,  # Not Modified
    307,  # Temporary Redirect
    308,  # Permanent Redirect
    400,  # Bad Request
    401,  # Unauthorized
    402,  # Payment Required
    403,  # Forbidden
    404,  # Not Found
    406,  # Not Acceptable
    408,  # Request Timeout
    409,  # Conflict
    413,  # Payload Too Large
    422,  # Unprocessable Entity
    424,  # Failed Dependency
    429,  # Too Many Requests
    498,  # Invalid Token
    499,  # Client Closed Request
    500,  # Internal Server Error
    501,  # Not Implemented
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    520,  # Web server is returning an unknown error
    522,  # Connection timed out
    524,  # A timeout occurred
    529,  # Site is overloaded
}

# Combine the sets
ALLOWED_NUMBERS = ALLOWED_NUMBERS.union(HTTP_STATUS_CODES)


class HardcodedNumberFinder(ast.NodeVisitor):
    def __init__(self):
        self.hardcoded_numbers = []

    def visit_Constant(self, node):
        # For Python 3.8+
        if isinstance(node.value, (int, float)) and node.value not in ALLOWED_NUMBERS:
            self.hardcoded_numbers.append((node.lineno, node.value))
        self.generic_visit(node)

    def visit_Num(self, node):
        # For older Python versions
        if node.n not in ALLOWED_NUMBERS:
            self.hardcoded_numbers.append((node.lineno, node.n))
        self.generic_visit(node)


def check_file(filename):
    try:
        with open(filename, "r") as f:
            content = f.read()

        tree = ast.parse(content)
        finder = HardcodedNumberFinder()
        finder.visit(tree)

        if finder.hardcoded_numbers:
            print(f"ERROR in {filename}: Hardcoded numbers detected:")
            for line, value in finder.hardcoded_numbers:
                print(f"  Line {line}: {value}")
            return 1
        return 0
    except SyntaxError:
        print(f"Syntax error in {filename}")
        return 0


def main():
    exit_code = 0
    folder = "../../litellm"
    ignore_files = [
        "constants.py",
        "proxy_cli.py",
        "token_counter.py",
        "mock_functions.py",
        "duration_parser.py",
        "utils.py",
    ]
    ignore_folder = "types"
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".py") and filename not in ignore_files:
                full_path = os.path.join(root, filename)
                if ignore_folder in full_path:
                    continue
                exit_code |= check_file(full_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
