# TODO(mabdinur): Remove the pymongoengine integration, this integration does nothing special
# it just uses the pymongo integration and creates unnecessary pin objects
import mongoengine

from ..pymongo.patch import patch as patch_pymongo_module
from ..pymongo.patch import unpatch as unpatch_pymongo_module
from .trace import WrappedConnect


# Original connect function
_connect = mongoengine.connect


def get_version():
    # type: () -> str
    return getattr(mongoengine, "__version__", "")


def patch():
    if getattr(mongoengine, "_datadog_patch", False):
        return
    mongoengine.connect = WrappedConnect(_connect)
    mongoengine._datadog_patch = True
    patch_pymongo_module()


def unpatch():
    if not getattr(mongoengine, "_datadog_patch", False):
        return
    mongoengine.connect = _connect
    mongoengine._datadog_patch = False
    unpatch_pymongo_module()
