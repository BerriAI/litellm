import pytest

from litellm.telemetry.records import StatusClass


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (None, StatusClass.NONE),
        (199, StatusClass.NONE),
        (200, StatusClass.SUCCESS),
        (299, StatusClass.SUCCESS),
        (301, StatusClass.REDIRECT),
        (400, StatusClass.CLIENT_ERROR),
        (429, StatusClass.CLIENT_ERROR),
        (500, StatusClass.SERVER_ERROR),
        (599, StatusClass.SERVER_ERROR),
        (600, StatusClass.NONE),
    ],
)
def test_status_code_maps_to_its_class(status_code: int | None, expected: StatusClass) -> None:
    assert StatusClass.from_status_code(status_code) is expected
