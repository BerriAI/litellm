"""Constants shared between TensorBoard command line uploader and SDK uploader"""

from tensorboard.plugins.distribution import (
    metadata as distribution_metadata,
)
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.plugins.histogram import (
    metadata as histogram_metadata,
)
from tensorboard.plugins.hparams import metadata as hparams_metadata
from tensorboard.plugins.image import metadata as images_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.plugins.text import metadata as text_metadata

ALLOWED_PLUGINS = [
    scalar_metadata.PLUGIN_NAME,
    histogram_metadata.PLUGIN_NAME,
    distribution_metadata.PLUGIN_NAME,
    text_metadata.PLUGIN_NAME,
    hparams_metadata.PLUGIN_NAME,
    images_metadata.PLUGIN_NAME,
    graphs_metadata.PLUGIN_NAME,
]

# Minimum length of a logdir polling cycle in seconds. Shorter cycles will
# sleep to avoid spinning over the logdir, which isn't great for disks and can
# be expensive for network file systems.
MIN_LOGDIR_POLL_INTERVAL_SECS = 1

# Maximum length of a base-128 varint as used to encode a 64-bit value
# (without the "msb of last byte is bit 63" optimization, to be
# compatible with protobuf and golang varints).
MAX_VARINT64_LENGTH_BYTES = 10

# Default minimum interval between initiating WriteTensorbordRunData RPCs in
# milliseconds.
DEFAULT_MIN_SCALAR_REQUEST_INTERVAL = 10

# Default maximum WriteTensorbordRunData request size in bytes.
DEFAULT_MAX_SCALAR_REQUEST_SIZE = 128 * (2**10)  # 128KiB

# Default minimum interval between initiating WriteTensorbordRunData RPCs in
# milliseconds.
DEFAULT_MIN_TENSOR_REQUEST_INTERVAL = 10

# Default minimum interval between initiating WriteTensorbordRunData RPCs in
# milliseconds.
DEFAULT_MIN_BLOB_REQUEST_INTERVAL = 10

# Default maximum WriteTensorbordRunData request size in bytes.
DEFAULT_MAX_TENSOR_REQUEST_SIZE = 512 * (2**10)  # 512KiB

DEFAULT_MAX_BLOB_REQUEST_SIZE = 128 * (2**10)  # 24KiB

# Default maximum tensor point size in bytes.
DEFAULT_MAX_TENSOR_POINT_SIZE = 16 * (2**10)  # 16KiB

DEFAULT_MAX_BLOB_SIZE = 10 * (2**30)  # 10GiB
