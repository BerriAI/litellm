import pytest

from litellm.repositories.performance_alert_repository import is_statement_timeout


def _raw_query_error(message: str) -> Exception:
    from prisma.errors import RawQueryError

    return RawQueryError({"user_facing_error": {"is_panic": False, "message": message, "meta": {}}})


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("Raw query failed. Code: `57014`. Message: `ERROR: canceling statement due to statement timeout`", True),
        ("Error querying the database: FATAL: sorry, too many clients already", False),
        (
            "Raw query failed. Code: `57P01`. Message: `FATAL: terminating connection due to administrator command`",
            False,
        ),
        ("Raw query failed. Code: `40P01`. Message: `ERROR: deadlock detected`", False),
    ),
)
def test_only_a_statement_timeout_marks_a_resource_unreadable(message: str, expected: bool) -> None:
    assert is_statement_timeout(_raw_query_error(message)) is expected
