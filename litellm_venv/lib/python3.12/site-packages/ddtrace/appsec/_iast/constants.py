from typing import Any
from typing import Dict


VULN_INSECURE_HASHING_TYPE = "WEAK_HASH"
VULN_WEAK_CIPHER_TYPE = "WEAK_CIPHER"
VULN_SQL_INJECTION = "SQL_INJECTION"
VULN_PATH_TRAVERSAL = "PATH_TRAVERSAL"
VULN_WEAK_RANDOMNESS = "WEAK_RANDOMNESS"
VULN_INSECURE_COOKIE = "INSECURE_COOKIE"
VULN_NO_HTTPONLY_COOKIE = "NO_HTTPONLY_COOKIE"
VULN_NO_SAMESITE_COOKIE = "NO_SAMESITE_COOKIE"
VULN_CMDI = "COMMAND_INJECTION"
VULN_HEADER_INJECTION = "HEADER_INJECTION"
VULN_SSRF = "SSRF"

VULNERABILITY_TOKEN_TYPE = Dict[int, Dict[str, Any]]

HEADER_NAME_VALUE_SEPARATOR = ": "

MD5_DEF = "md5"
SHA1_DEF = "sha1"

DES_DEF = "des"
BLOWFISH_DEF = "blowfish"
RC2_DEF = "rc2"
RC4_DEF = "rc4"
IDEA_DEF = "idea"

DEFAULT_WEAK_HASH_ALGORITHMS = {MD5_DEF, SHA1_DEF}

DEFAULT_WEAK_CIPHER_ALGORITHMS = {DES_DEF, BLOWFISH_DEF, RC2_DEF, RC4_DEF, IDEA_DEF}

DEFAULT_WEAK_RANDOMNESS_FUNCTIONS = {
    "random",
    "randint",
    "randrange",
    "choice",
    "shuffle",
    "betavariate",
    "gammavariate",
    "expovariate",
    "choices",
    "gauss",
    "uniform",
    "lognormvariate",
    "normalvariate",
    "paretovariate",
    "sample",
    "triangular",
    "vonmisesvariate",
    "weibullvariate",
    "randbytes",
}

DEFAULT_PATH_TRAVERSAL_FUNCTIONS = {
    "glob": {"glob"},
    "os": {
        "mkdir",
        "remove",
        "rename",
        "rmdir",
        "listdir",
    },
    "pickle": {"load"},
    "_pickle": {"load"},
    "posix": {
        "mkdir",
        "remove",
        "rename",
        "rmdir",
        "listdir",
    },
    "shutil": {
        "copy",
        "copytree",
        "move",
        "rmtree",
    },
    "tarfile": {"open"},
    "zipfile": {"ZipFile"},
}
DBAPI_SQLITE = "sqlite"
DBAPI_PSYCOPG = "psycopg"
DBAPI_MYSQL = "mysql"
DBAPI_MARIADB = "mariadb"
DBAPI_INTEGRATIONS = (DBAPI_SQLITE, DBAPI_PSYCOPG, DBAPI_MYSQL, DBAPI_MARIADB)
