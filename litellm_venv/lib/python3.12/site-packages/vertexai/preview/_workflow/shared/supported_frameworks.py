# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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
#

import importlib

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata
import inspect
import sys
from typing import Any, List, Tuple
import warnings

from google.cloud.aiplatform import base
from packaging import version


_LOGGER = base.Logger(__name__)

# This most likely needs to be map
REMOTE_FRAMEWORKS = frozenset(["sklearn", "keras", "lightning"])

REMOTE_TRAINING_MODEL_UPDATE_ONLY_OVERRIDE_LIST = frozenset(["fit", "train"])

# Methods that change the state of the object during a training workflow
REMOTE_TRAINING_STATEFUL_OVERRIDE_LIST = frozenset(["fit", "train", "fit_transform"])

# Methods that don't change the state of the object during a training workflow
REMOTE_TRAINING_FUNCTIONAL_OVERRIDE_LIST = frozenset(["transform"])

# Methods involved in training process
REMOTE_TRAINING_OVERRIDE_LIST = (
    REMOTE_TRAINING_STATEFUL_OVERRIDE_LIST | REMOTE_TRAINING_FUNCTIONAL_OVERRIDE_LIST
)

REMOTE_PREDICTION_OVERRIDE_LIST = frozenset(["predict"])

REMOTE_OVERRIDE_LIST = REMOTE_TRAINING_OVERRIDE_LIST.union(
    REMOTE_PREDICTION_OVERRIDE_LIST
)


LIBRARY_TO_MODULE_MAP = {"scikit-learn": "sklearn", "tf-models-official": "official"}


def _get_version_for_package(package_name: str) -> str:
    try:
        # Note: this doesn't work in the internal environment since
        # importlib.metadata relies on the directory site-packages to collect
        # the metadata of Python packages.
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        _LOGGER.info(
            "Didn't find package %s via importlib.metadata. Trying to import it.",
            package_name,
        )
    try:
        if package_name in LIBRARY_TO_MODULE_MAP:
            module_name = LIBRARY_TO_MODULE_MAP[package_name]
        else:
            # Note: this assumes the top-level module name is the same as the
            # package name after replacing "-" in the package name by "_".
            # This is not always true.
            module_name = package_name.replace("-", "_")

        module = importlib.import_module(module_name)
        # This assumes the top-level module has __version__ attribute, but this
        # is not always true.
        return module.__version__
    except Exception as exc:
        raise RuntimeError(f"{package_name} is not installed.") from exc


def _get_mro(cls_or_ins: Any) -> Tuple[Any, ...]:
    if inspect.isclass(cls_or_ins):
        return cls_or_ins.__mro__
    else:
        return cls_or_ins.__class__.__mro__


# pylint: disable=g-import-not-at-top
def _is_keras(cls_or_ins: Any) -> bool:
    try:
        global keras
        from tensorflow import keras

        return keras.layers.Layer in _get_mro(cls_or_ins)
    except ImportError:
        return False


def _is_sklearn(cls_or_ins: Any) -> bool:
    try:
        global sklearn
        import sklearn

        return sklearn.base.BaseEstimator in _get_mro(cls_or_ins)
    except ImportError:
        return False


def _is_lightning(cls_or_ins: Any) -> bool:
    try:
        global torch
        global lightning
        import torch
        import lightning

        return lightning.pytorch.trainer.trainer.Trainer in _get_mro(cls_or_ins)
    except ImportError:
        return False


def _is_torch(cls_or_ins: Any) -> bool:
    try:
        global torch
        import torch

        return torch.nn.modules.module.Module in _get_mro(cls_or_ins)
    except ImportError:
        return False


def _is_torch_dataloader(cls_or_ins: Any) -> bool:
    try:
        global torch
        import torch

        return torch.utils.data.DataLoader in _get_mro(cls_or_ins)
    except ImportError:
        return False


def _is_tensorflow(cls_or_ins: Any) -> bool:
    try:
        global tf
        import tensorflow as tf

        return tf.Module in _get_mro(cls_or_ins)
    except ImportError:
        return False


def _is_pandas_dataframe(possible_dataframe: Any) -> bool:
    try:
        global pd
        import pandas as pd

        return pd.DataFrame in _get_mro(possible_dataframe)
    except ImportError:
        return False


def _is_bigframe(possible_dataframe: Any) -> bool:
    try:
        global bf
        import bigframes as bf
        from bigframes.dataframe import DataFrame

        return DataFrame in _get_mro(possible_dataframe)
    except ImportError:
        return False


# pylint: enable=g-import-not-at-top
def _is_oss(cls_or_ins: Any) -> bool:
    return any(
        [_is_sklearn(cls_or_ins), _is_keras(cls_or_ins), _is_lightning(cls_or_ins)]
    )


# pylint: disable=undefined-variable
def _get_deps_if_sklearn_model(model: Any) -> List[str]:
    deps = []
    if _is_sklearn(model):
        dep_version = version.Version(sklearn.__version__).base_version
        deps.append(f"scikit-learn=={dep_version}")
    return deps


def _get_deps_if_tensorflow_model(model: Any) -> List[str]:
    deps = []
    if _is_tensorflow(model):
        dep_version = version.Version(tf.__version__).base_version
        deps.append(f"tensorflow=={dep_version}")
    return deps


def _get_deps_if_torch_model(model: Any) -> List[str]:
    deps = []
    if _is_torch(model):
        dep_version = version.Version(torch.__version__).base_version
        deps.append(f"torch=={dep_version}")
    return deps


def _get_deps_if_lightning_model(model: Any) -> List[str]:
    deps = []
    if _is_lightning(model):
        lightning_version = version.Version(lightning.__version__).base_version
        torch_version = version.Version(torch.__version__).base_version
        deps.append(f"lightning=={lightning_version}")
        deps.append(f"torch=={torch_version}")
        try:
            global tensorboard
            import tensorboard

            tensorboard_version = version.Version(tensorboard.__version__).base_version
            deps.append(f"tensorboard=={tensorboard_version}")
        except ImportError:
            pass
        try:
            global tensorboardX
            import tensorboardX

            tensorboardX_version = version.Version(
                tensorboardX.__version__
            ).base_version
            deps.append(f"tensorboardX=={tensorboardX_version}")
        except ImportError:
            pass

    return deps


def _get_deps_if_torch_dataloader(obj: Any) -> List[str]:
    deps = []
    if _is_torch_dataloader(obj):
        dep_version = version.Version(torch.__version__).base_version
        deps.append(f"torch=={dep_version}")
        deps.extend(_get_cloudpickle_deps())
    return deps


def _get_cloudpickle_deps() -> List[str]:
    deps = []
    try:
        global cloudpickle
        import cloudpickle

        dep_version = version.Version(cloudpickle.__version__).base_version
        deps.append(f"cloudpickle=={dep_version}")
    except ImportError as e:
        raise ImportError(
            "Cloudpickle is not installed. Please call `pip install google-cloud-aiplatform[preview]`."
        ) from e

    return deps


def _get_deps_if_pandas_dataframe(possible_dataframe: Any) -> List[str]:
    deps = []
    if _is_pandas_dataframe(possible_dataframe):
        dep_version = version.Version(pd.__version__).base_version
        deps.append(f"pandas=={dep_version}")
        deps += _get_pyarrow_deps()
    # Note: it's likely that a DataFrame can be changed to other format, and
    # therefore needs to be serialized by CloudPickleSerializer. An example
    # is sklearn's Transformer.fit_transform() method, whose output is always
    # a ndarray.
    deps += _get_cloudpickle_deps()
    return deps


def _get_pyarrow_deps() -> List[str]:
    deps = []
    try:
        global pyarrow
        import pyarrow

        dep_version = version.Version(pyarrow.__version__).base_version
        deps.append(f"pyarrow=={dep_version}")
    except ImportError:
        deps.append("pyarrow")
    return deps


def _get_numpy_deps() -> List[str]:
    deps = []
    try:
        global numpy
        import numpy

        dep_version = version.Version(numpy.__version__).base_version
        deps.append(f"numpy=={dep_version}")
    except ImportError:
        deps.append("numpy")
    return deps


def _get_pandas_deps() -> List[str]:
    deps = []
    try:
        global pd
        import pandas as pd

        dep_version = version.Version(pd.__version__).base_version
        deps.append(f"pandas=={dep_version}")
    except ImportError:
        deps.append("pandas")
    return deps


# pylint: enable=undefined-variable


def _get_estimator_requirement(estimator: Any) -> List[str]:
    """Returns a list of requirements given an estimator."""
    deps = []
    deps.extend(_get_numpy_deps())
    deps.extend(_get_pandas_deps())
    deps.extend(_get_cloudpickle_deps())
    deps.extend(_get_deps_if_sklearn_model(estimator))
    deps.extend(_get_deps_if_tensorflow_model(estimator))
    deps.extend(_get_deps_if_torch_model(estimator))
    deps.extend(_get_deps_if_lightning_model(estimator))
    # dedupe the dependencies by casting it to a dict first (dict perserves the
    # order while set doesn't)
    return list(dict.fromkeys(deps))


def _get_python_minor_version() -> str:
    # this will generally be the container with least or no security vulnerabilities
    return ".".join(sys.version.split()[0].split(".")[0:2])


def _get_cpu_container_uri() -> str:
    """Returns the container uri used for cpu training."""
    return f"python:{_get_python_minor_version()}"


def _get_gpu_container_uri(estimator: Any) -> str:
    """Returns the container uri used for gpu training given an estimator."""
    local_python_version = _get_python_minor_version()
    if _is_tensorflow(estimator):
        if local_python_version != "3.10":
            warnings.warn(
                f"Your local runtime has python{local_python_version}, but your "
                "remote GPU training will be executed in python3.10"
            )
        return "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11.py310:latest"

    elif _is_torch(estimator) or _is_lightning(estimator):
        if local_python_version != "3.10":
            warnings.warn(
                f"Your local runtime has python{local_python_version}, but your "
                "remote GPU training will be executed in python3.10"
            )
        return "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime"

    else:
        raise ValueError(f"{estimator} is not supported for GPU training.")
