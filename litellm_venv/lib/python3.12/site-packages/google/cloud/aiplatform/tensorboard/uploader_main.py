# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Launches Tensorboard Uploader for TB.GCP."""
import re

from absl import app
from absl import flags
from google.api_core import exceptions
from google.cloud import aiplatform
from google.cloud.aiplatform import jobs
from google.cloud.aiplatform.constants import base as constants
from google.cloud.aiplatform.tensorboard import uploader
from google.cloud.aiplatform.tensorboard import uploader_constants
from google.cloud.aiplatform.tensorboard import uploader_utils
from google.cloud.aiplatform.utils import TensorboardClientWithOverride


FLAGS = flags.FLAGS
flags.DEFINE_string("experiment_name", None, "The name of the Cloud AI Experiment.")
flags.DEFINE_string(
    "experiment_display_name", None, "The display name of the Cloud AI Experiment."
)
flags.DEFINE_string(
    "description", None, "String description to assign to the experiment."
)
flags.DEFINE_string("logdir", None, "Tensorboard log directory to upload")
flags.DEFINE_bool("one_shot", False, "Iterate through logdir once to upload.")
flags.DEFINE_string("env", "prod", "Environment which this tensorboard belongs to.")
flags.DEFINE_string(
    "tensorboard_resource_name",
    None,
    "Tensorboard resource to create this experiment in. ",
)
flags.DEFINE_integer(
    "event_file_inactive_secs",
    None,
    "Age in seconds of last write after which an event file is considered inactive.",
)
flags.DEFINE_string(
    "run_name_prefix",
    None,
    "If present, all runs created by this invocation will have their name "
    "prefixed by this value.",
)
flags.DEFINE_string(
    "api_uri",
    "aiplatform.googleapis.com",
    "The API URI for fetching Tensorboard metadata.",
)
flags.DEFINE_string(
    "web_server_uri",
    "tensorboard.googleusercontent.com",
    "The API URI for accessing the Tensorboard UI.",
)

flags.DEFINE_multi_string(
    "allowed_plugins",
    uploader_constants.ALLOWED_PLUGINS,
    "Plugins allowed by the Uploader.",
)

flags.mark_flags_as_required(["experiment_name", "logdir", "tensorboard_resource_name"])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    constants.API_BASE_PATH = FLAGS.api_uri
    m = re.match(
        "projects/(.*)/locations/(.*)/tensorboards/.*", FLAGS.tensorboard_resource_name
    )
    project_id = m[1]
    region = m[2]
    api_client = aiplatform.initializer.global_config.create_client(
        client_class=TensorboardClientWithOverride,
        location_override=region,
    )

    (
        blob_storage_bucket,
        blob_storage_folder,
    ) = uploader_utils.get_blob_storage_bucket_and_folder(
        api_client, FLAGS.tensorboard_resource_name, project_id
    )

    experiment_name = FLAGS.experiment_name
    experiment_display_name = get_experiment_display_name_with_override(
        experiment_name, FLAGS.experiment_display_name, project_id, region
    )

    tb_uploader = uploader.TensorBoardUploader(
        experiment_name=experiment_name,
        experiment_display_name=experiment_display_name,
        description=FLAGS.description,
        tensorboard_resource_name=FLAGS.tensorboard_resource_name,
        blob_storage_bucket=blob_storage_bucket,
        blob_storage_folder=blob_storage_folder,
        allowed_plugins=FLAGS.allowed_plugins,
        writer_client=api_client,
        logdir=FLAGS.logdir,
        one_shot=FLAGS.one_shot,
        event_file_inactive_secs=FLAGS.event_file_inactive_secs,
        run_name_prefix=FLAGS.run_name_prefix,
    )

    tb_uploader.create_experiment()

    print(
        "View your Tensorboard at https://{}.{}/experiment/{}".format(
            region,
            FLAGS.web_server_uri,
            tb_uploader.get_experiment_resource_name().replace("/", "+"),
        )
    )
    tb_uploader.start_uploading()


def get_experiment_display_name_with_override(
    experiment_name, experiment_display_name, project_id, region
):
    if experiment_name.isdecimal() and not experiment_display_name:
        try:
            return jobs.CustomJob.get(
                resource_name=experiment_name,
                project=project_id,
                location=region,
            ).display_name
        except exceptions.NotFound:
            return experiment_display_name
    return experiment_display_name


def flags_parser(args):
    # Plumbs the flags defined in this file to the main module, mostly for the
    # console script wrapper tb-gcp-uploader.
    for flag in set(flags.FLAGS.get_key_flags_for_module(__name__)):
        flags.FLAGS.register_flag_by_module(args[0], flag)
    return app.parse_flags_with_usage(args)


def run_main():
    app.run(main, flags_parser=flags_parser)


if __name__ == "__main__":
    run_main()
