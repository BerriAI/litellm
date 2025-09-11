# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from argparse import ArgumentParser

from huggingface_hub.cli.auth import AuthCommands
from huggingface_hub.cli.cache import CacheCommand
from huggingface_hub.cli.download import DownloadCommand
from huggingface_hub.cli.jobs import JobsCommands
from huggingface_hub.cli.lfs import LfsCommands
from huggingface_hub.cli.repo import RepoCommands
from huggingface_hub.cli.repo_files import RepoFilesCommand
from huggingface_hub.cli.system import EnvironmentCommand, VersionCommand
from huggingface_hub.cli.upload import UploadCommand
from huggingface_hub.cli.upload_large_folder import UploadLargeFolderCommand


def main():
    parser = ArgumentParser("hf", usage="hf <command> [<args>]")
    commands_parser = parser.add_subparsers(help="hf command helpers")

    # Register commands
    AuthCommands.register_subcommand(commands_parser)
    CacheCommand.register_subcommand(commands_parser)
    DownloadCommand.register_subcommand(commands_parser)
    JobsCommands.register_subcommand(commands_parser)
    RepoCommands.register_subcommand(commands_parser)
    RepoFilesCommand.register_subcommand(commands_parser)
    UploadCommand.register_subcommand(commands_parser)
    UploadLargeFolderCommand.register_subcommand(commands_parser)

    # System commands
    EnvironmentCommand.register_subcommand(commands_parser)
    VersionCommand.register_subcommand(commands_parser)

    # LFS commands (hidden in --help)
    LfsCommands.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    if service is not None:
        service.run()


if __name__ == "__main__":
    main()
