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
"""Training script to be run in Vertex CustomJob.
"""

# import modules
import os

from absl import app
from absl import flags
import vertexai
from vertexai.preview._workflow.serialization_engine import (
    any_serializer,
    serializers_base,
)
from vertexai.preview._workflow.shared import (
    constants,
    supported_frameworks,
    model_utils,
)
from vertexai.preview.developer import remote_specs


os.environ["_IS_VERTEX_REMOTE_TRAINING"] = "True"

print(constants._START_EXECUTION_MSG)

_ARGS = flags.DEFINE_list(
    "arg_names", [], "Argument names of those to be deserialized."
)
# TODO(b/274979556): consider other approaches to pass around the primitives
_PASS_THROUGH_INT_ARGS = flags.DEFINE_list(
    "pass_through_int_args", [], "Pass-through integer arguments."
)
_PASS_THROUGH_FLOAT_ARGS = flags.DEFINE_list(
    "pass_through_float_args", [], "Pass-through float arguments."
)
_PASS_THROUGH_BOOL_ARGS = flags.DEFINE_list(
    "pass_through_bool_args", [], "Pass-through bool arguments."
)
_PASS_THROUGH_STR_ARGS = flags.DEFINE_list(
    "pass_through_str_args", [], "Pass-through string arguments."
)
_METHOD_NAME = flags.DEFINE_string("method_name", None, "Method being called")

_INPUT_PATH = flags.DEFINE_string("input_path", None, "input path.")
_OUTPUT_PATH = flags.DEFINE_string("output_path", None, "output path.")
_ENABLE_AUTOLOG = flags.DEFINE_bool("enable_autolog", False, "enable autolog.")
_ENABLE_CUDA = flags.DEFINE_bool("enable_cuda", False, "enable cuda.")
_ENABLE_DISTRIBUTED = flags.DEFINE_bool(
    "enable_distributed", False, "enable distributed training."
)
_ACCELERATOR_COUNT = flags.DEFINE_integer(
    "accelerator_count",
    0,
    "accelerator count for single worker, multi-gpu training.",
)


# pylint: disable=protected-access
def main(argv):
    del argv

    # set cuda for tensorflow & pytorch
    try:
        import tensorflow

        if not _ENABLE_CUDA.value:
            tensorflow.config.set_visible_devices([], "GPU")
    except ImportError:
        pass

    try:
        import torch

        torch.set_default_device("cuda" if _ENABLE_CUDA.value else "cpu")
    except ImportError:
        torch = None

    strategy = None
    try:
        from tensorflow import keras  # noqa: F401

        # distribute strategy must be initialized at the beginning of the program
        # to avoid RuntimeError: "Collective ops must be configured at program startup"
        strategy = remote_specs._get_keras_distributed_strategy(
            _ENABLE_DISTRIBUTED.value, _ACCELERATOR_COUNT.value
        )

    except ImportError:
        pass

    if _ENABLE_AUTOLOG.value:
        vertexai.preview.init(autolog=True)

    # retrieve the estimator
    serializer = any_serializer.AnySerializer()
    # load the global metadata
    serializer.load_global_metadata(
        os.path.join(_INPUT_PATH.value, any_serializer.GLOBAL_SERIALIZATION_METADATA)
    )

    estimator = serializer.deserialize(
        os.path.join(_INPUT_PATH.value, "input_estimator")
    )

    if strategy and supported_frameworks._is_keras(estimator):
        # Single worker, multi-gpu will be compiled with tf.distribute.MirroredStrategy.
        # Multi-worker will be compiled with tf.distribute.MultiWorkerMirroredStrategy.
        # Single worker CPU/GPU will be returned as is.
        estimator = remote_specs._set_keras_distributed_strategy(estimator, strategy)

    if supported_frameworks._is_lightning(estimator):
        from lightning.pytorch.trainer.connectors.accelerator_connector import (
            _AcceleratorConnector,
        )

        # Re-instantiate accelerator connecotor in remote environment. Most of configs
        # like strategy, devices will be automatically handled by
        # the _AcceleratorConnector class.
        # accelerator and num_nodes need to be manually set.
        accelerator = "gpu" if _ENABLE_CUDA.value else "cpu"
        num_nodes = (
            remote_specs._get_cluster_spec().get_world_size()
            if _ENABLE_DISTRIBUTED.value
            else 1
        )
        estimator._accelerator_connector = _AcceleratorConnector(
            accelerator=accelerator,
            num_nodes=num_nodes,
        )

    # retrieve seriliazed_args
    kwargs = {}
    for arg_name in _ARGS.value:
        arg_value = serializer.deserialize(os.path.join(_INPUT_PATH.value, arg_name))

        if supported_frameworks._is_torch_dataloader(arg_value):
            # update gpu setting in dataloader for pytorch model gpu training
            # lightning will automatically handle the data so no need to update
            if supported_frameworks._is_torch(estimator) and _ENABLE_CUDA.value:
                arg_value.pin_memory = True
                arg_value.pin_memory_device = "cuda"
                arg_value.generator = torch.Generator("cuda")
                if hasattr(arg_value.sampler, "generator"):
                    setattr(arg_value.sampler, "generator", arg_value.generator)
            # make sure the torch default device is the same as
            # dataloader generator's device
            torch.set_default_device(
                arg_value.generator.device.type if arg_value.generator else "cpu"
            )

        kwargs[arg_name] = arg_value

    for arg_name_and_arg_value in _PASS_THROUGH_INT_ARGS.value:
        arg_name, arg_value = arg_name_and_arg_value.split("=")
        kwargs[arg_name] = int(arg_value)
    for arg_name_and_arg_value in _PASS_THROUGH_FLOAT_ARGS.value:
        arg_name, arg_value = arg_name_and_arg_value.split("=")
        kwargs[arg_name] = float(arg_value)
    for arg_name_and_arg_value in _PASS_THROUGH_BOOL_ARGS.value:
        arg_name, arg_value = arg_name_and_arg_value.split("=")
        kwargs[arg_name] = bool(arg_value)
    for arg_name_and_arg_value in _PASS_THROUGH_STR_ARGS.value:
        arg_name, arg_value = arg_name_and_arg_value.split("=")
        kwargs[arg_name] = arg_value

    # for all custom trainers, set cluster_spec if available
    if (
        isinstance(estimator, vertexai.preview.VertexModel)
        and _ENABLE_DISTRIBUTED.value
    ):
        setattr(estimator, "cluster_spec", remote_specs._get_cluster_spec())
        if supported_frameworks._is_torch(estimator):
            # need to know if GPU training is enabled for the
            # optional remote_specs.setup_pytorch_distributed_training()
            # function that a user can call in train()
            setattr(estimator, "_enable_cuda", _ENABLE_CUDA.value)

    output = getattr(estimator, _METHOD_NAME.value)(**kwargs)

    # serialize the output
    os.makedirs(_OUTPUT_PATH.value, exist_ok=True)

    if (
        _METHOD_NAME.value
        in supported_frameworks.REMOTE_TRAINING_STATEFUL_OVERRIDE_LIST
    ):
        # for distributed training, chief saves output to specified output
        # directory while non-chief workers save output to temp directory.
        output_path = remote_specs._get_output_path_for_distributed_training(
            _OUTPUT_PATH.value, model_utils._OUTPUT_ESTIMATOR_DIR
        )
        serializer.serialize(estimator, output_path)

        # for pytorch lightning trainer, we want to serialize the trained model as well
        if "model" in _ARGS.value:
            serializer.serialize(kwargs["model"], os.path.join(output_path, "model"))

    # for remote prediction
    if _METHOD_NAME.value in supported_frameworks.REMOTE_PREDICTION_OVERRIDE_LIST:
        serializer.serialize(
            output,
            os.path.join(_OUTPUT_PATH.value, model_utils._OUTPUT_PREDICTIONS_DIR),
        )

    output_path = remote_specs._get_output_path_for_distributed_training(
        _OUTPUT_PATH.value, "output_data"
    )
    try:
        serializer.serialize(output, output_path)
    except serializers_base.SerializationError as e:
        print(f"failed to serialize the output due to {e}")
    serializer.save_global_metadata(
        os.path.join(_OUTPUT_PATH.value, any_serializer.GLOBAL_SERIALIZATION_METADATA)
    )

    print(constants._END_EXECUTION_MSG)


if __name__ == "__main__":
    app.run(main)
