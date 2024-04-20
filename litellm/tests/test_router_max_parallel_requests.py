# What is this?
## Unit tests for the max_parallel_requests feature on Router
import sys, os, time, inspect, asyncio, traceback
from datetime import datetime
import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.utils import calculate_max_parallel_requests

"""
- only rpm
- only tpm
- only max_parallel_requests 
- max_parallel_requests + rpm 
- max_parallel_requests + tpm
- max_parallel_requests + tpm + rpm 
"""


max_parallel_requests_values = [None, 10]
tpm_values = [None, 20]
rpm_values = [None, 30]
default_max_parallel_requests = [None, 40]


@pytest.mark.parametrize(
    "max_parallel_requests, tpm, rpm, default_max_parallel_requests",
    [
        (mp, tp, rp, dmp)
        for mp in max_parallel_requests_values
        for tp in tpm_values
        for rp in rpm_values
        for dmp in default_max_parallel_requests
    ],
)
def test_scenario(max_parallel_requests, tpm, rpm, default_max_parallel_requests):
    calculated_max_parallel_requests = calculate_max_parallel_requests(
        max_parallel_requests=max_parallel_requests,
        rpm=rpm,
        tpm=tpm,
        default_max_parallel_requests=default_max_parallel_requests,
    )
    if max_parallel_requests is not None:
        assert max_parallel_requests == calculated_max_parallel_requests
    elif rpm is not None:
        assert rpm == calculated_max_parallel_requests
    elif tpm is not None:
        assert int(tpm / 1000 / 6) == calculated_max_parallel_requests
    elif default_max_parallel_requests is not None:
        assert calculated_max_parallel_requests == default_max_parallel_requests
    else:
        assert calculated_max_parallel_requests is None
