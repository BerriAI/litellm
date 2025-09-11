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
"""Contains commands to authenticate to the Hugging Face Hub and interact with your repositories.

Usage:
    # login and save token locally.
    hf auth login --token=hf_*** --add-to-git-credential

    # switch between tokens
    hf auth switch

    # list all tokens
    hf auth list

    # logout from all tokens
    hf auth logout

    # check which account you are logged in as
    hf auth whoami
"""

from argparse import _SubParsersAction
from typing import List, Optional

from requests.exceptions import HTTPError

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.hf_api import HfApi

from .._login import auth_list, auth_switch, login, logout
from ..utils import get_stored_tokens, get_token, logging
from ._cli_utils import ANSI


logger = logging.get_logger(__name__)

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice

    _inquirer_py_available = True
except ImportError:
    _inquirer_py_available = False


class AuthCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        # Create the main 'auth' command
        auth_parser = parser.add_parser(
            "auth", help="Manage authentication (login, logout, etc.)."
        )
        auth_subparsers = auth_parser.add_subparsers(help="Authentication subcommands")

        # Show help if no subcommand is provided
        auth_parser.set_defaults(func=lambda args: auth_parser.print_help())

        # Add 'login' as a subcommand of 'auth'
        login_parser = auth_subparsers.add_parser(
            "login", help="Log in using a token from huggingface.co/settings/tokens"
        )
        login_parser.add_argument(
            "--token",
            type=str,
            help="Token generated from https://huggingface.co/settings/tokens",
        )
        login_parser.add_argument(
            "--add-to-git-credential",
            action="store_true",
            help="Optional: Save token to git credential helper.",
        )
        login_parser.set_defaults(func=lambda args: AuthLogin(args))

        # Add 'logout' as a subcommand of 'auth'
        logout_parser = auth_subparsers.add_parser("logout", help="Log out")
        logout_parser.add_argument(
            "--token-name",
            type=str,
            help="Optional: Name of the access token to log out from.",
        )
        logout_parser.set_defaults(func=lambda args: AuthLogout(args))

        # Add 'whoami' as a subcommand of 'auth'
        whoami_parser = auth_subparsers.add_parser(
            "whoami", help="Find out which huggingface.co account you are logged in as."
        )
        whoami_parser.set_defaults(func=lambda args: AuthWhoami(args))

        # Existing subcommands
        auth_switch_parser = auth_subparsers.add_parser(
            "switch", help="Switch between access tokens"
        )
        auth_switch_parser.add_argument(
            "--token-name",
            type=str,
            help="Optional: Name of the access token to switch to.",
        )
        auth_switch_parser.add_argument(
            "--add-to-git-credential",
            action="store_true",
            help="Optional: Save token to git credential helper.",
        )
        auth_switch_parser.set_defaults(func=lambda args: AuthSwitch(args))

        auth_list_parser = auth_subparsers.add_parser(
            "list", help="List all stored access tokens"
        )
        auth_list_parser.set_defaults(func=lambda args: AuthList(args))


class BaseAuthCommand:
    def __init__(self, args):
        self.args = args
        self._api = HfApi()


class AuthLogin(BaseAuthCommand):
    def run(self):
        logging.set_verbosity_info()
        login(
            token=self.args.token,
            add_to_git_credential=self.args.add_to_git_credential,
        )


class AuthLogout(BaseAuthCommand):
    def run(self):
        logging.set_verbosity_info()
        logout(token_name=self.args.token_name)


class AuthSwitch(BaseAuthCommand):
    def run(self):
        logging.set_verbosity_info()
        token_name = self.args.token_name
        if token_name is None:
            token_name = self._select_token_name()

        if token_name is None:
            print("No token name provided. Aborting.")
            exit()
        auth_switch(token_name, add_to_git_credential=self.args.add_to_git_credential)

    def _select_token_name(self) -> Optional[str]:
        token_names = list(get_stored_tokens().keys())

        if not token_names:
            logger.error("No stored tokens found. Please login first.")
            return None

        if _inquirer_py_available:
            return self._select_token_name_tui(token_names)
        # if inquirer is not available, use a simpler terminal UI
        print("Available stored tokens:")
        for i, token_name in enumerate(token_names, 1):
            print(f"{i}. {token_name}")
        while True:
            try:
                choice = input(
                    "Enter the number of the token to switch to (or 'q' to quit): "
                )
                if choice.lower() == "q":
                    return None
                index = int(choice) - 1
                if 0 <= index < len(token_names):
                    return token_names[index]
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

    def _select_token_name_tui(self, token_names: List[str]) -> Optional[str]:
        choices = [Choice(token_name, name=token_name) for token_name in token_names]
        try:
            return inquirer.select(
                message="Select a token to switch to:",
                choices=choices,
                default=None,
            ).execute()
        except KeyboardInterrupt:
            logger.info("Token selection cancelled.")
            return None


class AuthList(BaseAuthCommand):
    def run(self):
        logging.set_verbosity_info()
        auth_list()


class AuthWhoami(BaseAuthCommand):
    def run(self):
        token = get_token()
        if token is None:
            print("Not logged in")
            exit()
        try:
            info = self._api.whoami(token)
            print(info["name"])
            orgs = [org["name"] for org in info["orgs"]]
            if orgs:
                print(ANSI.bold("orgs: "), ",".join(orgs))

            if ENDPOINT != "https://huggingface.co":
                print(f"Authenticated through private endpoint: {ENDPOINT}")
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
