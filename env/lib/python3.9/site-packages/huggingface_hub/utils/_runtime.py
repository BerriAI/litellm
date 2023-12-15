# coding=utf-8
# Copyright 2022-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check presence of installed packages at runtime."""
import platform
import sys
from typing import Any, Dict

import packaging.version

from .. import __version__, constants


_PY_VERSION: str = sys.version.split()[0].rstrip("+")

if packaging.version.Version(_PY_VERSION) < packaging.version.Version("3.8.0"):
    import importlib_metadata  # type: ignore
else:
    import importlib.metadata as importlib_metadata  # type: ignore


_package_versions = {}

_CANDIDATES = {
    "aiohttp": {"aiohttp"},
    "fastai": {"fastai"},
    "fastcore": {"fastcore"},
    "gradio": {"gradio"},
    "graphviz": {"graphviz"},
    "hf_transfer": {"hf_transfer"},
    "jinja": {"Jinja2"},
    "numpy": {"numpy"},
    "pillow": {"Pillow"},
    "pydantic": {"pydantic"},
    "pydot": {"pydot"},
    "tensorboard": {"tensorboardX"},
    "tensorflow": (
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-gpu",
        "tf-nightly",
        "tf-nightly-cpu",
        "tf-nightly-gpu",
        "intel-tensorflow",
        "intel-tensorflow-avx512",
        "tensorflow-rocm",
        "tensorflow-macos",
    ),
    "torch": {"torch"},
}

# Check once at runtime
for candidate_name, package_names in _CANDIDATES.items():
    _package_versions[candidate_name] = "N/A"
    for name in package_names:
        try:
            _package_versions[candidate_name] = importlib_metadata.version(name)
            break
        except importlib_metadata.PackageNotFoundError:
            pass


def _get_version(package_name: str) -> str:
    return _package_versions.get(package_name, "N/A")


def _is_available(package_name: str) -> bool:
    return _get_version(package_name) != "N/A"


# Python
def get_python_version() -> str:
    return _PY_VERSION


# Huggingface Hub
def get_hf_hub_version() -> str:
    return __version__


# aiohttp
def is_aiohttp_available() -> bool:
    return _is_available("aiohttp")


def get_aiohttp_version() -> str:
    return _get_version("aiohttp")


# FastAI
def is_fastai_available() -> bool:
    return _is_available("fastai")


def get_fastai_version() -> str:
    return _get_version("fastai")


# Fastcore
def is_fastcore_available() -> bool:
    return _is_available("fastcore")


def get_fastcore_version() -> str:
    return _get_version("fastcore")


# FastAI
def is_gradio_available() -> bool:
    return _is_available("gradio")


def get_gradio_version() -> str:
    return _get_version("gradio")


# Graphviz
def is_graphviz_available() -> bool:
    return _is_available("graphviz")


def get_graphviz_version() -> str:
    return _get_version("graphviz")


# hf_transfer
def is_hf_transfer_available() -> bool:
    return _is_available("hf_transfer")


def get_hf_transfer_version() -> str:
    return _get_version("hf_transfer")


# Numpy
def is_numpy_available() -> bool:
    return _is_available("numpy")


def get_numpy_version() -> str:
    return _get_version("numpy")


# Jinja
def is_jinja_available() -> bool:
    return _is_available("jinja")


def get_jinja_version() -> str:
    return _get_version("jinja")


# Pillow
def is_pillow_available() -> bool:
    return _is_available("pillow")


def get_pillow_version() -> str:
    return _get_version("pillow")


# Pydantic
def is_pydantic_available() -> bool:
    return _is_available("pydantic")


def get_pydantic_version() -> str:
    return _get_version("pydantic")


# Pydot
def is_pydot_available() -> bool:
    return _is_available("pydot")


def get_pydot_version() -> str:
    return _get_version("pydot")


# Tensorboard
def is_tensorboard_available() -> bool:
    return _is_available("tensorboard")


def get_tensorboard_version() -> str:
    return _get_version("tensorboard")


# Tensorflow
def is_tf_available() -> bool:
    return _is_available("tensorflow")


def get_tf_version() -> str:
    return _get_version("tensorflow")


# Torch
def is_torch_available() -> bool:
    return _is_available("torch")


def get_torch_version() -> str:
    return _get_version("torch")


# Shell-related helpers
try:
    # Set to `True` if script is running in a Google Colab notebook.
    # If running in Google Colab, git credential store is set globally which makes the
    # warning disappear. See https://github.com/huggingface/huggingface_hub/issues/1043
    #
    # Taken from https://stackoverflow.com/a/63519730.
    _is_google_colab = "google.colab" in str(get_ipython())  # type: ignore # noqa: F821
except NameError:
    _is_google_colab = False


def is_notebook() -> bool:
    """Return `True` if code is executed in a notebook (Jupyter, Colab, QTconsole).

    Taken from https://stackoverflow.com/a/39662359.
    Adapted to make it work with Google colab as well.
    """
    try:
        shell_class = get_ipython().__class__  # type: ignore # noqa: F821
        for parent_class in shell_class.__mro__:  # e.g. "is subclass of"
            if parent_class.__name__ == "ZMQInteractiveShell":
                return True  # Jupyter notebook, Google colab or qtconsole
        return False
    except NameError:
        return False  # Probably standard Python interpreter


def is_google_colab() -> bool:
    """Return `True` if code is executed in a Google colab.

    Taken from https://stackoverflow.com/a/63519730.
    """
    return _is_google_colab


def dump_environment_info() -> Dict[str, Any]:
    """Dump information about the machine to help debugging issues.

    Similar helper exist in:
    - `datasets` (https://github.com/huggingface/datasets/blob/main/src/datasets/commands/env.py)
    - `diffusers` (https://github.com/huggingface/diffusers/blob/main/src/diffusers/commands/env.py)
    - `transformers` (https://github.com/huggingface/transformers/blob/main/src/transformers/commands/env.py)
    """
    from huggingface_hub import HfFolder, whoami
    from huggingface_hub.utils import list_credential_helpers

    token = HfFolder().get_token()

    # Generic machine info
    info: Dict[str, Any] = {
        "huggingface_hub version": get_hf_hub_version(),
        "Platform": platform.platform(),
        "Python version": get_python_version(),
    }

    # Interpreter info
    try:
        shell_class = get_ipython().__class__  # type: ignore # noqa: F821
        info["Running in iPython ?"] = "Yes"
        info["iPython shell"] = shell_class.__name__
    except NameError:
        info["Running in iPython ?"] = "No"
    info["Running in notebook ?"] = "Yes" if is_notebook() else "No"
    info["Running in Google Colab ?"] = "Yes" if is_google_colab() else "No"

    # Login info
    info["Token path ?"] = HfFolder().path_token
    info["Has saved token ?"] = token is not None
    if token is not None:
        try:
            info["Who am I ?"] = whoami()["name"]
        except Exception:
            pass

    try:
        info["Configured git credential helpers"] = ", ".join(list_credential_helpers())
    except Exception:
        pass

    # Installed dependencies
    info["FastAI"] = get_fastai_version()
    info["Tensorflow"] = get_tf_version()
    info["Torch"] = get_torch_version()
    info["Jinja2"] = get_jinja_version()
    info["Graphviz"] = get_graphviz_version()
    info["Pydot"] = get_pydot_version()
    info["Pillow"] = get_pillow_version()
    info["hf_transfer"] = get_hf_transfer_version()
    info["gradio"] = get_gradio_version()
    info["tensorboard"] = get_tensorboard_version()
    info["numpy"] = get_numpy_version()
    info["pydantic"] = get_pydantic_version()
    info["aiohttp"] = get_aiohttp_version()

    # Environment variables
    info["ENDPOINT"] = constants.ENDPOINT
    info["HUGGINGFACE_HUB_CACHE"] = constants.HUGGINGFACE_HUB_CACHE
    info["HUGGINGFACE_ASSETS_CACHE"] = constants.HUGGINGFACE_ASSETS_CACHE
    info["HF_TOKEN_PATH"] = constants.HF_TOKEN_PATH
    info["HF_HUB_OFFLINE"] = constants.HF_HUB_OFFLINE
    info["HF_HUB_DISABLE_TELEMETRY"] = constants.HF_HUB_DISABLE_TELEMETRY
    info["HF_HUB_DISABLE_PROGRESS_BARS"] = constants.HF_HUB_DISABLE_PROGRESS_BARS
    info["HF_HUB_DISABLE_SYMLINKS_WARNING"] = constants.HF_HUB_DISABLE_SYMLINKS_WARNING
    info["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = constants.HF_HUB_DISABLE_EXPERIMENTAL_WARNING
    info["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = constants.HF_HUB_DISABLE_IMPLICIT_TOKEN
    info["HF_HUB_ENABLE_HF_TRANSFER"] = constants.HF_HUB_ENABLE_HF_TRANSFER

    print("\nCopy-and-paste the text below in your GitHub issue.\n")
    print("\n".join([f"- {prop}: {val}" for prop, val in info.items()]) + "\n")
    return info
