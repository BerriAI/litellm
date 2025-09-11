"""Async executor versions of file functions from the os module."""

import os

from . import ospath as path
from .ospath import wrap

__all__ = [
    "path",
    "stat",
    "rename",
    "renames",
    "replace",
    "remove",
    "unlink",
    "mkdir",
    "makedirs",
    "rmdir",
    "removedirs",
    "symlink",
    "readlink",
    "listdir",
    "scandir",
    "access",
    "wrap",
    "getcwd",
]
if hasattr(os, "link"):
    __all__ += ["link"]
if hasattr(os, "sendfile"):
    __all__ += ["sendfile"]
if hasattr(os, "statvfs"):
    __all__ += ["statvfs"]


stat = wrap(os.stat)
rename = wrap(os.rename)
renames = wrap(os.renames)
replace = wrap(os.replace)
remove = wrap(os.remove)
unlink = wrap(os.unlink)
mkdir = wrap(os.mkdir)
makedirs = wrap(os.makedirs)
rmdir = wrap(os.rmdir)
removedirs = wrap(os.removedirs)
symlink = wrap(os.symlink)
readlink = wrap(os.readlink)
listdir = wrap(os.listdir)
scandir = wrap(os.scandir)
access = wrap(os.access)
getcwd = wrap(os.getcwd)

if hasattr(os, "link"):
    link = wrap(os.link)
if hasattr(os, "sendfile"):
    sendfile = wrap(os.sendfile)
if hasattr(os, "statvfs"):
    statvfs = wrap(os.statvfs)
