# JSONDecodeError was introduced in requests=2.27 released in 2022.
# This allows us to support older requests for users
# More information: https://github.com/psf/requests/pull/5856
try:
    from requests import JSONDecodeError  # type: ignore  # noqa: F401
except ImportError:
    try:
        from simplejson import JSONDecodeError  # type: ignore # noqa: F401
    except ImportError:
        from json import JSONDecodeError  # type: ignore  # noqa: F401

import contextlib
import os
import shutil
import stat
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Optional, Union

import yaml


# Wrap `yaml.dump` to set `allow_unicode=True` by default.
#
# Example:
# ```py
# >>> yaml.dump({"emoji": "👀", "some unicode": "日本か"})
# 'emoji: "\\U0001F440"\nsome unicode: "\\u65E5\\u672C\\u304B"\n'
#
# >>> yaml_dump({"emoji": "👀", "some unicode": "日本か"})
# 'emoji: "👀"\nsome unicode: "日本か"\n'
# ```
yaml_dump: Callable[..., str] = partial(yaml.dump, stream=None, allow_unicode=True)  # type: ignore


@contextlib.contextmanager
def SoftTemporaryDirectory(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[Union[Path, str]] = None,
    **kwargs,
) -> Generator[str, None, None]:
    """
    Context manager to create a temporary directory and safely delete it.

    If tmp directory cannot be deleted normally, we set the WRITE permission and retry.
    If cleanup still fails, we give up but don't raise an exception. This is equivalent
    to  `tempfile.TemporaryDirectory(..., ignore_cleanup_errors=True)` introduced in
    Python 3.10.

    See https://www.scivision.dev/python-tempfile-permission-error-windows/.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix=prefix, suffix=suffix, dir=dir, **kwargs)
    yield tmpdir.name

    try:
        # First once with normal cleanup
        shutil.rmtree(tmpdir.name)
    except Exception:
        # If failed, try to set write permission and retry
        try:
            shutil.rmtree(tmpdir.name, onerror=_set_write_permission_and_retry)
        except Exception:
            pass

    # And finally, cleanup the tmpdir.
    # If it fails again, give up but do not throw error
    try:
        tmpdir.cleanup()
    except Exception:
        pass


def _set_write_permission_and_retry(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)
