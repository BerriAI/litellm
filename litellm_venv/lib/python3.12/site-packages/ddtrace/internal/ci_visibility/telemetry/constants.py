from enum import Enum


CIVISIBILITY_TELEMETRY_NAMESPACE = "civisibility"


class ERROR_TYPES(str, Enum):
    TIMEOUT = "timeout"
    NETWORK = "network"
    CODE_4XX = "status_code_4xx_response"
    CODE_5XX = "status_code_5xx_response"
    BAD_JSON = "bad_json"
    UNKNOWN = "unknown"


class TEST_FRAMEWORKS(str, Enum):
    PYTEST = "pytest"
    UNITTEST = "unittest"
    MANUAL = "unset"


class EVENT_TYPES(str, Enum):
    SESSION = "session"
    MODULE = "module"
    SUITE = "suite"
    TEST = "test"
    UNSET = "unset"


class GIT_TELEMETRY_COMMANDS(str, Enum):
    GET_REPOSITORY = "get_repository"
    GET_BRANCH = "get_branch"
    CHECK_SHALLOW = "check_shallow"
    UNSHALLOW = "unshallow"
    GET_LOCAL_COMMITS = "get_local_commits"
    GET_OBJECTS = "get_objects"
    PACK_OBJECTS = "pack_objects"


class GIT_TELEMETRY(str, Enum):
    COMMAND_COUNT = "git.command"
    COMMAND_MS = "git.command_ms"
    COMMAND_ERRORS = "git.command_errors"

    SEARCH_COMMITS_COUNT = "git_requests.search_commits"
    SEARCH_COMMITS_MS = "git_requests.search_commits_ms"
    SEARCH_COMMITS_ERRORS = "git_requests.search_commits_errors"

    OBJECTS_PACK_COUNT = "git_requests.objects_pack"
    OBJECTS_PACK_MS = "git_requests.objects_pack_ms"
    OBJECTS_PACK_ERRORS = "git_requests.objects_pack_errors"
    OBJECTS_PACK_FILES = "git_requests.objects_pack_files"
    OBJECTS_PACK_BYTES = "git_requests.objects_pack_bytes"

    SETTINGS_COUNT = "git_requests.settings"
    SETTINGS_MS = "git_requests.settings_ms"
    SETTINGS_ERRORS = "git_requests.settings_errors"
    SETTINGS_RESPONSE = "git_requests.settings_response"
