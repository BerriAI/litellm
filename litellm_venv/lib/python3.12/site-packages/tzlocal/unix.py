import logging
import os
import re
import sys
import warnings
from datetime import timezone

from tzlocal import utils

import zoneinfo

_cache_tz = None
_cache_tz_name = None

log = logging.getLogger("tzlocal")


def _get_localzone_name(_root="/"):
    """Tries to find the local timezone configuration.

    This method finds the timezone name, if it can, or it returns None.

    The parameter _root makes the function look for files like /etc/localtime
    beneath the _root directory. This is primarily used by the tests.
    In normal usage you call the function without parameters."""

    # First try the ENV setting.
    tzenv = utils._tz_name_from_env()
    if tzenv:
        return tzenv

    # Are we under Termux on Android?
    if os.path.exists(os.path.join(_root, "system/bin/getprop")):
        log.debug("This looks like Termux")

        import subprocess

        try:
            androidtz = (
                subprocess.check_output(["getprop", "persist.sys.timezone"])
                .strip()
                .decode()
            )
            return androidtz
        except (OSError, subprocess.CalledProcessError):
            # proot environment or failed to getprop
            log.debug("It's not termux?")
            pass

    # Now look for distribution specific configuration files
    # that contain the timezone name.

    # Stick all of them in a dict, to compare later.
    found_configs = {}

    for configfile in ("etc/timezone", "var/db/zoneinfo"):
        tzpath = os.path.join(_root, configfile)
        try:
            with open(tzpath) as tzfile:
                data = tzfile.read()
                log.debug(f"{tzpath} found, contents:\n {data}")

                etctz = data.strip("/ \t\r\n")
                if not etctz:
                    # Empty file, skip
                    continue
                for etctz in etctz.splitlines():
                    # Get rid of host definitions and comments:
                    if " " in etctz:
                        etctz, dummy = etctz.split(" ", 1)
                    if "#" in etctz:
                        etctz, dummy = etctz.split("#", 1)
                    if not etctz:
                        continue

                    found_configs[tzpath] = etctz.replace(" ", "_")

        except (OSError, UnicodeDecodeError):
            # File doesn't exist or is a directory, or it's a binary file.
            continue

    # CentOS has a ZONE setting in /etc/sysconfig/clock,
    # OpenSUSE has a TIMEZONE setting in /etc/sysconfig/clock and
    # Gentoo has a TIMEZONE setting in /etc/conf.d/clock
    # We look through these files for a timezone:

    zone_re = re.compile(r"\s*ZONE\s*=\s*\"")
    timezone_re = re.compile(r"\s*TIMEZONE\s*=\s*\"")
    end_re = re.compile('"')

    for filename in ("etc/sysconfig/clock", "etc/conf.d/clock"):
        tzpath = os.path.join(_root, filename)
        try:
            with open(tzpath, "rt") as tzfile:
                data = tzfile.readlines()
                log.debug(f"{tzpath} found, contents:\n {data}")

            for line in data:
                # Look for the ZONE= setting.
                match = zone_re.match(line)
                if match is None:
                    # No ZONE= setting. Look for the TIMEZONE= setting.
                    match = timezone_re.match(line)
                if match is not None:
                    # Some setting existed
                    line = line[match.end() :]
                    etctz = line[: end_re.search(line).start()]

                    # We found a timezone
                    found_configs[tzpath] = etctz.replace(" ", "_")

        except (OSError, UnicodeDecodeError):
            # UnicodeDecode handles when clock is symlink to /etc/localtime
            continue

    # systemd distributions use symlinks that include the zone name,
    # see manpage of localtime(5) and timedatectl(1)
    tzpath = os.path.join(_root, "etc/localtime")
    if os.path.exists(tzpath) and os.path.islink(tzpath):
        log.debug(f"{tzpath} found")
        etctz = os.path.realpath(tzpath)
        start = etctz.find("/") + 1
        while start != 0:
            etctz = etctz[start:]
            try:
                zoneinfo.ZoneInfo(etctz)
                tzinfo = f"{tzpath} is a symlink to"
                found_configs[tzinfo] = etctz.replace(" ", "_")
                # Only need first valid relative path in simlink.
                break
            except zoneinfo.ZoneInfoNotFoundError:
                pass
            start = etctz.find("/") + 1

    if len(found_configs) > 0:
        log.debug(f"{len(found_configs)} found:\n {found_configs}")

        # We found some explicit config of some sort!
        if len(found_configs) > 1:
            # Uh-oh, multiple configs. See if they match:
            unique_tzs = _get_unique_tzs(found_configs, _root)

            if len(unique_tzs) != 1 and "etc/timezone" in str(found_configs.keys()):
                # For some reason some distros are removing support for /etc/timezone,
                # which is bad, because that's the only place where the timezone is stated
                # in plain text, and what's worse, they don't delete it. So we can't trust
                # it now, so when we have conflicting configs, we just ignore it, with a warning.
                log.warning(
                    "/etc/timezone is deprecated in some distros, and no longer reliable. "
                    "tzlocal is ignoring it, and you can likely delete it."
                )
                found_configs = {
                    k: v for k, v in found_configs.items() if "etc/timezone" not in k
                }
                unique_tzs = _get_unique_tzs(found_configs, _root)

            if len(unique_tzs) != 1:
                message = "Multiple conflicting time zone configurations found:\n"
                for key, value in found_configs.items():
                    message += f"{key}: {value}\n"
                message += "Fix the configuration, or set the time zone in a TZ environment variable.\n"
                raise zoneinfo.ZoneInfoNotFoundError(message)

        # We found exactly one config! Use it.
        return list(found_configs.values())[0]


def _get_unique_tzs(found_configs, _root):
    unique_tzs = set()
    zoneinfopath = os.path.join(_root, "usr", "share", "zoneinfo")
    directory_depth = len(zoneinfopath.split(os.path.sep))

    for tzname in found_configs.values():
        # Look them up in /usr/share/zoneinfo, and find what they
        # really point to:
        path = os.path.realpath(os.path.join(zoneinfopath, *tzname.split("/")))
        real_zone_name = "/".join(path.split(os.path.sep)[directory_depth:])
        unique_tzs.add(real_zone_name)

    return unique_tzs


def _get_localzone(_root="/"):
    """Creates a timezone object from the timezone name.

    If there is no timezone config, it will try to create a file from the
    localtime timezone, and if there isn't one, it will default to UTC.

    The parameter _root makes the function look for files like /etc/localtime
    beneath the _root directory. This is primarily used by the tests.
    In normal usage you call the function without parameters."""

    # First try the ENV setting.
    tzenv = utils._tz_from_env()
    if tzenv:
        return tzenv

    tzname = _get_localzone_name(_root)
    if tzname is None:
        # No explicit setting existed. Use localtime
        log.debug("No explicit setting existed. Use localtime")
        for filename in ("etc/localtime", "usr/local/etc/localtime"):
            tzpath = os.path.join(_root, filename)

            if not os.path.exists(tzpath):
                continue
            with open(tzpath, "rb") as tzfile:
                tz = zoneinfo.ZoneInfo.from_file(tzfile, key="local")
                break
        else:
            warnings.warn("Can not find any timezone configuration, defaulting to UTC.")
            utcname = [x for x in zoneinfo.available_timezones() if "UTC" in x]
            if utcname:
                tz = zoneinfo.ZoneInfo(utcname[0])
            else:
                tz = timezone.utc
    else:
        tz = zoneinfo.ZoneInfo(tzname)

    if _root == "/":
        # We are using a file in etc to name the timezone.
        # Verify that the timezone specified there is actually used:
        utils.assert_tz_offset(tz, error=False)
    return tz


def get_localzone_name() -> str:
    """Get the computers configured local timezone name, if any."""
    global _cache_tz_name
    if _cache_tz_name is None:
        _cache_tz_name = _get_localzone_name()

    return _cache_tz_name


def get_localzone() -> zoneinfo.ZoneInfo:
    """Get the computers configured local timezone, if any."""

    global _cache_tz
    if _cache_tz is None:
        _cache_tz = _get_localzone()

    return _cache_tz


def reload_localzone() -> zoneinfo.ZoneInfo:
    """Reload the cached localzone. You need to call this if the timezone has changed."""
    global _cache_tz_name
    global _cache_tz
    _cache_tz_name = _get_localzone_name()
    _cache_tz = _get_localzone()

    return _cache_tz
