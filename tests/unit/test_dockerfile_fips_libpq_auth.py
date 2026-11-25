import re
from pathlib import Path
from typing import Final

DOCKERFILE_FIPS: Final = Path(__file__).resolve().parents[2] / "litellm/deploy/docker/Dockerfile.fips"


def test_fips_runtime_stage_refuses_md5_postgres_auth():
    dockerfile: Final = DOCKERFILE_FIPS.read_text(encoding="utf-8")
    runtime: Final = re.search(r"^FROM \S+ AS runtime\n(.*?)^FROM ", dockerfile, re.MULTILINE | re.DOTALL)
    assert runtime, "Dockerfile.fips has no `FROM ... AS runtime` stage"
    assert re.search(r'^ENV PGREQUIREAUTH="!md5"$', runtime.group(1), re.MULTILINE), (
        'The FIPS runtime stage must set PGREQUIREAUTH="!md5" so libpq refuses MD5 password authentication, '
        "whose hash would come from OpenSSL's legacy provider instead of the FIPS provider"
    )
