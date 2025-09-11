# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Contains commands to interact with repositories on the Hugging Face Hub.

Usage:
    # create a new dataset repo on the Hub
    hf repo create my-cool-dataset --repo-type=dataset

    # create a private model repo on the Hub
    hf repo create my-cool-model --private
"""

import argparse
from argparse import _SubParsersAction
from typing import Optional

from requests.exceptions import HTTPError

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.commands._cli_utils import ANSI
from huggingface_hub.constants import REPO_TYPES, SPACES_SDK_TYPES
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import logging


logger = logging.get_logger(__name__)


class RepoCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        repo_parser = parser.add_parser("repo", help="Manage repos on the Hub.")
        repo_subparsers = repo_parser.add_subparsers(
            help="huggingface.co repos related commands"
        )

        # Show help if no subcommand is provided
        repo_parser.set_defaults(func=lambda args: repo_parser.print_help())

        # CREATE
        repo_create_parser = repo_subparsers.add_parser(
            "create", help="Create a new repo on huggingface.co"
        )
        repo_create_parser.add_argument(
            "repo_id",
            type=str,
            help="The ID of the repo to create to (e.g. `username/repo-name`). The username is optional and will be set to your username if not provided.",
        )
        repo_create_parser.add_argument(
            "--repo-type",
            type=str,
            help='Optional: set to "dataset" or "space" if creating a dataset or space, default is model.',
        )
        repo_create_parser.add_argument(
            "--space_sdk",
            type=str,
            help='Optional: Hugging Face Spaces SDK type. Required when --type is set to "space".',
            choices=SPACES_SDK_TYPES,
        )
        repo_create_parser.add_argument(
            "--private",
            action="store_true",
            help="Whether to create a private repository. Defaults to public unless the organization's default is private.",
        )
        repo_create_parser.add_argument(
            "--token",
            type=str,
            help="Hugging Face token. Will default to the locally saved token if not provided.",
        )
        repo_create_parser.add_argument(
            "--exist-ok",
            action="store_true",
            help="Do not raise an error if repo already exists.",
        )
        repo_create_parser.add_argument(
            "--resource-group-id",
            type=str,
            help="Resource group in which to create the repo. Resource groups is only available for Enterprise Hub organizations.",
        )
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))

        # TAG SUBCOMMANDS
        repo_tag_parser = repo_subparsers.add_parser(
            "tag", help="Manage tags for a repo on the Hub."
        )
        tag_subparsers = repo_tag_parser.add_subparsers(
            help="Tag actions", dest="tag_action", required=True
        )

        # tag create
        tag_create_parser = tag_subparsers.add_parser(
            "create", help="Create a tag for a repo."
        )
        tag_create_parser.add_argument(
            "repo_id",
            type=str,
            help="The ID of the repo to tag (e.g. `username/repo-name`).",
        )
        tag_create_parser.add_argument(
            "tag", type=str, help="The name of the tag to create."
        )
        tag_create_parser.add_argument(
            "-m", "--message", type=str, help="The description of the tag to create."
        )
        tag_create_parser.add_argument(
            "--revision", type=str, help="The git revision to tag."
        )
        tag_create_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens.",
        )
        tag_create_parser.add_argument(
            "--repo-type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Set the type of repository (model, dataset, or space).",
        )
        tag_create_parser.set_defaults(func=lambda args: RepoTagCreateCommand(args))

        # tag list
        tag_list_parser = tag_subparsers.add_parser(
            "list", help="List tags for a repo."
        )
        tag_list_parser.add_argument(
            "repo_id",
            type=str,
            help="The ID of the repo to list tags for (e.g. `username/repo-name`).",
        )
        tag_list_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens.",
        )
        tag_list_parser.add_argument(
            "--repo-type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Set the type of repository (model, dataset, or space).",
        )
        tag_list_parser.set_defaults(func=lambda args: RepoTagListCommand(args))

        # tag delete
        tag_delete_parser = tag_subparsers.add_parser(
            "delete", help="Delete a tag from a repo."
        )
        tag_delete_parser.add_argument(
            "repo_id",
            type=str,
            help="The ID of the repo to delete the tag from (e.g. `username/repo-name`).",
        )
        tag_delete_parser.add_argument(
            "tag", type=str, help="The name of the tag to delete."
        )
        tag_delete_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Answer Yes to prompts automatically.",
        )
        tag_delete_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens.",
        )
        tag_delete_parser.add_argument(
            "--repo-type",
            choices=["model", "dataset", "space"],
            default="model",
            help="Set the type of repository (model, dataset, or space).",
        )
        tag_delete_parser.set_defaults(func=lambda args: RepoTagDeleteCommand(args))


class RepoCreateCommand:
    def __init__(self, args: argparse.Namespace):
        self.repo_id: str = args.repo_id
        self.repo_type: Optional[str] = args.repo_type
        self.space_sdk: Optional[str] = args.space_sdk
        self.private: bool = args.private
        self.token: Optional[str] = args.token
        self.exist_ok: bool = args.exist_ok
        self.resource_group_id: Optional[str] = args.resource_group_id
        self._api = HfApi()

    def run(self):
        repo_url = self._api.create_repo(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            private=self.private,
            token=self.token,
            exist_ok=self.exist_ok,
            resource_group_id=self.resource_group_id,
            space_sdk=self.space_sdk,
        )
        print(f"Successfully created {ANSI.bold(repo_url.repo_id)} on the Hub.")
        print(f"Your repo is now available at {ANSI.bold(repo_url)}")


class RepoTagCommand:
    def __init__(self, args):
        self.args = args
        self.api = HfApi(token=getattr(args, "token", None))
        self.repo_id = args.repo_id
        self.repo_type = getattr(args, "repo_type", "model")
        if self.repo_type not in REPO_TYPES:
            print("Invalid repo --repo-type")
            exit(1)


class RepoTagCreateCommand(RepoTagCommand):
    def run(self):
        print(
            f"You are about to create tag {ANSI.bold(self.args.tag)} on {self.repo_type} {ANSI.bold(self.repo_id)}"
        )
        try:
            self.api.create_tag(
                repo_id=self.repo_id,
                tag=self.args.tag,
                tag_message=getattr(self.args, "message", None),
                revision=getattr(self.args, "revision", None),
                repo_type=self.repo_type,
            )
        except RepositoryNotFoundError:
            print(f"{self.repo_type.capitalize()} {ANSI.bold(self.repo_id)} not found.")
            exit(1)
        except RevisionNotFoundError:
            print(
                f"Revision {ANSI.bold(getattr(self.args, 'revision', None))} not found."
            )
            exit(1)
        except HfHubHTTPError as e:
            if e.response.status_code == 409:
                print(
                    f"Tag {ANSI.bold(self.args.tag)} already exists on {ANSI.bold(self.repo_id)}"
                )
                exit(1)
            raise e
        print(f"Tag {ANSI.bold(self.args.tag)} created on {ANSI.bold(self.repo_id)}")


class RepoTagListCommand(RepoTagCommand):
    def run(self):
        try:
            refs = self.api.list_repo_refs(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
        except RepositoryNotFoundError:
            print(f"{self.repo_type.capitalize()} {ANSI.bold(self.repo_id)} not found.")
            exit(1)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        if len(refs.tags) == 0:
            print("No tags found")
            exit(0)
        print(f"Tags for {self.repo_type} {ANSI.bold(self.repo_id)}:")
        for tag in refs.tags:
            print(tag.name)


class RepoTagDeleteCommand(RepoTagCommand):
    def run(self):
        print(
            f"You are about to delete tag {ANSI.bold(self.args.tag)} on {self.repo_type} {ANSI.bold(self.repo_id)}"
        )
        if not getattr(self.args, "yes", False):
            choice = input("Proceed? [Y/n] ").lower()
            if choice not in ("", "y", "yes"):
                print("Abort")
                exit()
        try:
            self.api.delete_tag(
                repo_id=self.repo_id, tag=self.args.tag, repo_type=self.repo_type
            )
        except RepositoryNotFoundError:
            print(f"{self.repo_type.capitalize()} {ANSI.bold(self.repo_id)} not found.")
            exit(1)
        except RevisionNotFoundError:
            print(
                f"Tag {ANSI.bold(self.args.tag)} not found on {ANSI.bold(self.repo_id)}"
            )
            exit(1)
        print(f"Tag {ANSI.bold(self.args.tag)} deleted on {ANSI.bold(self.repo_id)}")
