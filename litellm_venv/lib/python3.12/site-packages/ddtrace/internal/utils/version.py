import typing  # noqa:F401
from typing import Optional  # noqa:F401

import ddtrace.vendor.packaging.version as packaging_version
from ddtrace.version import get_version


def parse_version(version):
    # type: (str) -> typing.Tuple[int, int, int]
    """Convert a version string to a tuple of (major, minor, micro)

    Examples::

       1.2.3           -> (1, 2, 3)
       1.2             -> (1, 2, 0)
       1               -> (1, 0, 0)
       1.0.0-beta1     -> (1, 0, 0)
       2020.6.19       -> (2020, 6, 19)
       malformed       -> (0, 0, 0)
       10.5.0 extra    -> (10, 5, 0)
    """
    # If we have any spaces/extra text, grab the first part
    #   "1.0.0 beta1" -> "1.0.0"
    #   "1.0.0" -> "1.0.0"
    # DEV: Versions with spaces will get converted to LegacyVersion, we do this splitting
    # to maximize the chances of getting a Version as a parsing result
    if " " in version:
        version = version.split()[0]

    # version() will not raise an exception, if the version if malformed instead
    # we will end up with a LegacyVersion

    try:
        parsed = packaging_version.parse(version)
    except packaging_version.InvalidVersion:
        # packaging>=22.0 raises an InvalidVersion instead of returning a LegacyVersion
        return (0, 0, 0)

    # LegacyVersion.release will always be `None`
    if not parsed.release:
        return (0, 0, 0)

    # Version.release was added in 17.1
    # packaging >= 20.0 has `Version.{major,minor,micro}`, use the following
    # to support older versions of the library
    # https://github.com/pypa/packaging/blob/47d40f640fddb7c97b01315419b6a1421d2dedbb/packaging/version.py#L404-L417
    return (
        parsed.release[0] if len(parsed.release) >= 1 else 0,
        parsed.release[1] if len(parsed.release) >= 2 else 0,
        parsed.release[2] if len(parsed.release) >= 3 else 0,
    )


def _pep440_to_semver(version=None):
    # type: (Optional[str]) -> str
    # The library uses a PEP 440-compliant (https://peps.python.org/pep-0440/) versioning
    # scheme, but the Agent spec requires that we use a SemVer-compliant version.
    #
    # However, we may have versions like:
    #
    #   - 1.7.1.dev3+gf258c7d9
    #   - 1.7.1rc2.dev3+gf258c7d9
    #
    # Which are not Semver-compliant.
    #
    # The easiest fix is to replace the first occurrence of "rc" or
    # ".dev" with "-rc" or "-dev" to make them compliant.
    #
    # Other than X.Y.Z, we are allowed `-<dot separated pre-release>+<build identifier>`
    # https://semver.org/#backusnaur-form-grammar-for-valid-semver-versions
    #
    # e.g. 1.7.1-rc2.dev3+gf258c7d9 is valid

    tracer_version = version or get_version()
    if "rc" in tracer_version and "-rc" not in tracer_version:
        tracer_version = tracer_version.replace("rc", "-rc", 1)
    elif ".dev" in tracer_version:
        tracer_version = tracer_version.replace(".dev", "-dev", 1)
    return tracer_version


version = _pep440_to_semver()
