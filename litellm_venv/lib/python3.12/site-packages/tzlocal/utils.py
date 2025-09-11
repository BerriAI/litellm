import calendar
import datetime
import logging
import os
import time
import warnings
import zoneinfo

from tzlocal import windows_tz

log = logging.getLogger("tzlocal")


def get_tz_offset(tz):
    """Get timezone's offset using built-in function datetime.utcoffset()."""
    return int(datetime.datetime.now(tz).utcoffset().total_seconds())


def assert_tz_offset(tz, error=True):
    """Assert that system's timezone offset equals to the timezone offset found.

    If they don't match, we probably have a misconfiguration, for example, an
    incorrect timezone set in /etc/timezone file in systemd distributions.

    If error is True, this method will raise a ValueError, otherwise it will
    emit a warning.
    """

    tz_offset = get_tz_offset(tz)
    system_offset = calendar.timegm(time.localtime()) - calendar.timegm(time.gmtime())
    # No one has timezone offsets less than a minute, so this should be close enough:
    if abs(tz_offset - system_offset) > 60:
        msg = (
            f"Timezone offset does not match system offset: {tz_offset} != {system_offset}. "
            "Please, check your config files."
        )
        if error:
            raise ValueError(msg)
        warnings.warn(msg)


def _tz_name_from_env(tzenv=None):
    if tzenv is None:
        tzenv = os.environ.get("TZ")

    if not tzenv:
        return None

    log.debug(f"Found a TZ environment: {tzenv}")

    if tzenv[0] == ":":
        tzenv = tzenv[1:]

    if tzenv in windows_tz.tz_win:
        # Yup, it's a timezone
        return tzenv

    if os.path.isabs(tzenv) and os.path.exists(tzenv):
        # It's a file specification, expand it, if possible
        parts = os.path.realpath(tzenv).split(os.sep)

        # Is it a zone info zone?
        possible_tz = "/".join(parts[-2:])
        if possible_tz in windows_tz.tz_win:
            # Yup, it is
            return possible_tz

        # Maybe it's a short one, like UTC?
        if parts[-1] in windows_tz.tz_win:
            # Indeed
            return parts[-1]

    log.debug("TZ does not contain a time zone name")
    return None


def _tz_from_env(tzenv=None):
    if tzenv is None:
        tzenv = os.environ.get("TZ")

    if not tzenv:
        return None

    # Some weird format that exists:
    if tzenv[0] == ":":
        tzenv = tzenv[1:]

    # TZ specifies a file
    if os.path.isabs(tzenv) and os.path.exists(tzenv):
        # Try to see if we can figure out the name
        tzname = _tz_name_from_env(tzenv)
        if not tzname:
            # Nope, not a standard timezone name, just take the filename
            tzname = tzenv.split(os.sep)[-1]
        with open(tzenv, "rb") as tzfile:
            return zoneinfo.ZoneInfo.from_file(tzfile, key=tzname)

    # TZ must specify a zoneinfo zone.
    try:
        tz = zoneinfo.ZoneInfo(tzenv)
        # That worked, so we return this:
        return tz
    except zoneinfo.ZoneInfoNotFoundError:
        # Nope, it's something like "PST4DST" etc, we can't handle that.
        raise zoneinfo.ZoneInfoNotFoundError(
            f"tzlocal() does not support non-zoneinfo timezones like {tzenv}. \n"
            "Please use a timezone in the form of Continent/City"
        ) from None
