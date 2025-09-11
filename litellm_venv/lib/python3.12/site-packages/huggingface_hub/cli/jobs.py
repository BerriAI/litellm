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
"""Contains commands to interact with jobs on the Hugging Face Hub.

Usage:
    # run a job
    hf jobs run <image> <command>

    # List running or completed jobs
    hf jobs ps [-a] [-f key=value] [--format TEMPLATE]

    # Stream logs from a job
    hf jobs logs <job-id>

    # Inspect detailed information about a job
    hf jobs inspect <job-id>

    # Cancel a running job
    hf jobs cancel <job-id>
"""

import json
import os
import re
from argparse import Namespace, _SubParsersAction
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

from huggingface_hub import HfApi, SpaceHardware, get_token
from huggingface_hub.utils import logging
from huggingface_hub.utils._dotenv import load_dotenv

from . import BaseHuggingfaceCLICommand


logger = logging.get_logger(__name__)

SUGGESTED_FLAVORS = [item.value for item in SpaceHardware if item.value != "zero-a10g"]


class JobsCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        jobs_parser = parser.add_parser("jobs", help="Run and manage Jobs on the Hub.")
        jobs_subparsers = jobs_parser.add_subparsers(
            help="huggingface.co jobs related commands"
        )

        # Show help if no subcommand is provided
        jobs_parser.set_defaults(func=lambda args: jobs_parser.print_help())

        # Register commands
        InspectCommand.register_subcommand(jobs_subparsers)
        LogsCommand.register_subcommand(jobs_subparsers)
        PsCommand.register_subcommand(jobs_subparsers)
        RunCommand.register_subcommand(jobs_subparsers)
        CancelCommand.register_subcommand(jobs_subparsers)
        UvCommand.register_subcommand(jobs_subparsers)


class RunCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("run", help="Run a Job")
        run_parser.add_argument("image", type=str, help="The Docker image to use.")
        run_parser.add_argument(
            "-e",
            "--env",
            action="append",
            help="Set environment variables. E.g. --env ENV=value",
        )
        run_parser.add_argument(
            "-s",
            "--secrets",
            action="append",
            help=(
                "Set secret environment variables. E.g. --secrets SECRET=value "
                "or `--secrets HF_TOKEN` to pass your Hugging Face token."
            ),
        )
        run_parser.add_argument(
            "--env-file", type=str, help="Read in a file of environment variables."
        )
        run_parser.add_argument(
            "--secrets-file",
            type=str,
            help="Read in a file of secret environment variables.",
        )
        run_parser.add_argument(
            "--flavor",
            type=str,
            help=f"Flavor for the hardware, as in HF Spaces. Defaults to `cpu-basic`. Possible values: {', '.join(SUGGESTED_FLAVORS)}.",
        )
        run_parser.add_argument(
            "--timeout",
            type=str,
            help="Max duration: int/float with s (seconds, default), m (minutes), h (hours) or d (days).",
        )
        run_parser.add_argument(
            "-d",
            "--detach",
            action="store_true",
            help="Run the Job in the background and print the Job ID.",
        )
        run_parser.add_argument(
            "--namespace",
            type=str,
            help="The namespace where the Job will be created. Defaults to the current user's namespace.",
        )
        run_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        run_parser.add_argument("command", nargs="...", help="The command to run.")
        run_parser.set_defaults(func=RunCommand)

    def __init__(self, args: Namespace) -> None:
        self.image: str = args.image
        self.command: List[str] = args.command
        self.env: dict[str, Optional[str]] = {}
        if args.env_file:
            self.env.update(
                load_dotenv(Path(args.env_file).read_text(), environ=os.environ.copy())
            )
        for env_value in args.env or []:
            self.env.update(load_dotenv(env_value, environ=os.environ.copy()))
        self.secrets: dict[str, Optional[str]] = {}
        extended_environ = _get_extended_environ()
        if args.secrets_file:
            self.secrets.update(
                load_dotenv(
                    Path(args.secrets_file).read_text(), environ=extended_environ
                )
            )
        for secret in args.secrets or []:
            self.secrets.update(load_dotenv(secret, environ=extended_environ))
        self.flavor: Optional[SpaceHardware] = args.flavor
        self.timeout: Optional[str] = args.timeout
        self.detach: bool = args.detach
        self.namespace: Optional[str] = args.namespace
        self.token: Optional[str] = args.token

    def run(self) -> None:
        api = HfApi(token=self.token)
        job = api.run_job(
            image=self.image,
            command=self.command,
            env=self.env,
            secrets=self.secrets,
            flavor=self.flavor,
            timeout=self.timeout,
            namespace=self.namespace,
        )
        # Always print the job ID to the user
        print(f"Job started with ID: {job.id}")
        print(f"View at: {job.url}")

        if self.detach:
            return

        # Now let's stream the logs
        for log in api.fetch_job_logs(job_id=job.id):
            print(log)


class LogsCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("logs", help="Fetch the logs of a Job")
        run_parser.add_argument("job_id", type=str, help="Job ID")
        run_parser.add_argument(
            "--namespace",
            type=str,
            help="The namespace where the job is running. Defaults to the current user's namespace.",
        )
        run_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        run_parser.set_defaults(func=LogsCommand)

    def __init__(self, args: Namespace) -> None:
        self.job_id: str = args.job_id
        self.namespace: Optional[str] = args.namespace
        self.token: Optional[str] = args.token

    def run(self) -> None:
        api = HfApi(token=self.token)
        for log in api.fetch_job_logs(job_id=self.job_id, namespace=self.namespace):
            print(log)


def _tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    terminal_width = max(os.get_terminal_size().columns, len(headers) * 12)
    while len(headers) + sum(col_widths) > terminal_width:
        col_to_minimize = col_widths.index(max(col_widths))
        col_widths[col_to_minimize] //= 2
        if len(headers) + sum(col_widths) <= terminal_width:
            col_widths[col_to_minimize] = (
                terminal_width
                - sum(col_widths)
                - len(headers)
                + col_widths[col_to_minimize]
            )
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        row_format_args = [
            str(x)[: col_width - 3] + "..." if len(str(x)) > col_width else str(x)
            for x, col_width in zip(row, col_widths)
        ]
        lines.append(row_format.format(*row_format_args))
    return "\n".join(lines)


class PsCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("ps", help="List Jobs")
        run_parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="Show all Jobs (default shows just running)",
        )
        run_parser.add_argument(
            "--namespace",
            type=str,
            help="The namespace from where it lists the jobs. Defaults to the current user's namespace.",
        )
        run_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        # Add Docker-style filtering argument
        run_parser.add_argument(
            "-f",
            "--filter",
            action="append",
            default=[],
            help="Filter output based on conditions provided (format: key=value)",
        )
        # Add option to format output
        run_parser.add_argument(
            "--format",
            type=str,
            help="Format output using a custom template",
        )
        run_parser.set_defaults(func=PsCommand)

    def __init__(self, args: Namespace) -> None:
        self.all: bool = args.all
        self.namespace: Optional[str] = args.namespace
        self.token: Optional[str] = args.token
        self.format: Optional[str] = args.format
        self.filters: Dict[str, str] = {}

        # Parse filter arguments (key=value pairs)
        for f in args.filter:
            if "=" in f:
                key, value = f.split("=", 1)
                self.filters[key.lower()] = value
            else:
                print(
                    f"Warning: Ignoring invalid filter format '{f}'. Use key=value format."
                )

    def run(self) -> None:
        """
        Fetch and display job information for the current user.
        Uses Docker-style filtering with -f/--filter flag and key=value pairs.
        """
        try:
            api = HfApi(token=self.token)

            # Fetch jobs data
            jobs = api.list_jobs(namespace=self.namespace)

            # Define table headers
            table_headers = ["JOB ID", "IMAGE/SPACE", "COMMAND", "CREATED", "STATUS"]

            # Process jobs data
            rows = []

            for job in jobs:
                # Extract job data for filtering
                status = job.status.stage if job.status else "UNKNOWN"

                # Skip job if not all jobs should be shown and status doesn't match criteria
                if not self.all and status not in ("RUNNING", "UPDATING"):
                    continue

                # Extract job ID
                job_id = job.id

                # Extract image or space information
                image_or_space = job.docker_image or "N/A"

                # Extract and format command
                command = job.command or []
                command_str = " ".join(command) if command else "N/A"

                # Extract creation time
                created_at = job.created_at or "N/A"

                # Create a dict with all job properties for filtering
                job_properties = {
                    "id": job_id,
                    "image": image_or_space,
                    "status": status.lower(),
                    "command": command_str,
                }

                # Check if job matches all filters
                if not self._matches_filters(job_properties):
                    continue

                # Create row
                rows.append([job_id, image_or_space, command_str, created_at, status])

            # Handle empty results
            if not rows:
                filters_msg = ""
                if self.filters:
                    filters_msg = f" matching filters: {', '.join([f'{k}={v}' for k, v in self.filters.items()])}"

                print(f"No jobs found{filters_msg}")
                return

            # Apply custom format if provided or use default tabular format
            self._print_output(rows, table_headers)

        except requests.RequestException as e:
            print(f"Error fetching jobs data: {e}")
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error processing jobs data: {e}")
        except Exception as e:
            print(f"Unexpected error - {type(e).__name__}: {e}")

    def _matches_filters(self, job_properties: Dict[str, str]) -> bool:
        """Check if job matches all specified filters."""
        for key, pattern in self.filters.items():
            # Check if property exists
            if key not in job_properties:
                return False

            # Support pattern matching with wildcards
            if "*" in pattern or "?" in pattern:
                # Convert glob pattern to regex
                regex_pattern = pattern.replace("*", ".*").replace("?", ".")
                if not re.search(
                    f"^{regex_pattern}$", job_properties[key], re.IGNORECASE
                ):
                    return False
            # Simple substring matching
            elif pattern.lower() not in job_properties[key].lower():
                return False

        return True

    def _print_output(self, rows, headers):
        """Print output according to the chosen format."""
        if self.format:
            # Custom template formatting (simplified)
            template = self.format
            for row in rows:
                line = template
                for i, field in enumerate(
                    ["id", "image", "command", "created", "status"]
                ):
                    placeholder = f"{{{{.{field}}}}}"
                    if placeholder in line:
                        line = line.replace(placeholder, str(row[i]))
                print(line)
        else:
            # Default tabular format
            print(
                _tabulate(
                    rows,
                    headers=headers,
                )
            )


class InspectCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser(
            "inspect", help="Display detailed information on one or more Jobs"
        )
        run_parser.add_argument(
            "--namespace",
            type=str,
            help="The namespace where the job is running. Defaults to the current user's namespace.",
        )
        run_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        run_parser.add_argument("job_ids", nargs="...", help="The jobs to inspect")
        run_parser.set_defaults(func=InspectCommand)

    def __init__(self, args: Namespace) -> None:
        self.namespace: Optional[str] = args.namespace
        self.token: Optional[str] = args.token
        self.job_ids: List[str] = args.job_ids

    def run(self) -> None:
        api = HfApi(token=self.token)
        jobs = [
            api.inspect_job(job_id=job_id, namespace=self.namespace)
            for job_id in self.job_ids
        ]
        print(json.dumps([asdict(job) for job in jobs], indent=4, default=str))


class CancelCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction) -> None:
        run_parser = parser.add_parser("cancel", help="Cancel a Job")
        run_parser.add_argument("job_id", type=str, help="Job ID")
        run_parser.add_argument(
            "--namespace",
            type=str,
            help="The namespace where the job is running. Defaults to the current user's namespace.",
        )
        run_parser.add_argument(
            "--token",
            type=str,
            help="A User Access Token generated from https://huggingface.co/settings/tokens",
        )
        run_parser.set_defaults(func=CancelCommand)

    def __init__(self, args: Namespace) -> None:
        self.job_id: str = args.job_id
        self.namespace = args.namespace
        self.token: Optional[str] = args.token

    def run(self) -> None:
        api = HfApi(token=self.token)
        api.cancel_job(job_id=self.job_id, namespace=self.namespace)


class UvCommand(BaseHuggingfaceCLICommand):
    """Run UV scripts on Hugging Face infrastructure."""

    @staticmethod
    def register_subcommand(parser):
        """Register UV run subcommand."""
        uv_parser = parser.add_parser(
            "uv",
            help="Run UV scripts (Python with inline dependencies) on HF infrastructure",
        )

        subparsers = uv_parser.add_subparsers(
            dest="uv_command", help="UV commands", required=True
        )

        # Run command only
        run_parser = subparsers.add_parser(
            "run",
            help="Run a UV script (local file or URL) on HF infrastructure",
        )
        run_parser.add_argument("script", help="UV script to run (local file or URL)")
        run_parser.add_argument(
            "script_args", nargs="...", help="Arguments for the script", default=[]
        )
        run_parser.add_argument(
            "--image", type=str, help="Use a custom Docker image with `uv` installed."
        )
        run_parser.add_argument(
            "--repo",
            help="Repository name for the script (creates ephemeral if not specified)",
        )
        run_parser.add_argument(
            "--flavor",
            type=str,
            help=f"Flavor for the hardware, as in HF Spaces. Defaults to `cpu-basic`. Possible values: {', '.join(SUGGESTED_FLAVORS)}.",
        )
        run_parser.add_argument(
            "-e", "--env", action="append", help="Environment variables"
        )
        run_parser.add_argument(
            "-s",
            "--secrets",
            action="append",
            help=(
                "Set secret environment variables. E.g. --secrets SECRET=value "
                "or `--secrets HF_TOKEN` to pass your Hugging Face token."
            ),
        )
        run_parser.add_argument(
            "--env-file", type=str, help="Read in a file of environment variables."
        )
        run_parser.add_argument(
            "--secrets-file",
            type=str,
            help="Read in a file of secret environment variables.",
        )
        run_parser.add_argument(
            "--timeout", type=str, help="Max duration (e.g., 30s, 5m, 1h)"
        )
        run_parser.add_argument(
            "-d", "--detach", action="store_true", help="Run in background"
        )
        run_parser.add_argument(
            "--namespace",
            type=str,
            help="The namespace where the Job will be created. Defaults to the current user's namespace.",
        )
        run_parser.add_argument("--token", type=str, help="HF token")
        # UV options
        run_parser.add_argument(
            "--with",
            action="append",
            help="Run with the given packages installed",
            dest="with_",
        )
        run_parser.add_argument(
            "-p",
            "--python",
            type=str,
            help="The Python interpreter to use for the run environment",
        )
        run_parser.set_defaults(func=UvCommand)

    def __init__(self, args: Namespace) -> None:
        """Initialize the command with parsed arguments."""
        self.script = args.script
        self.script_args = args.script_args
        self.dependencies = args.with_
        self.python = args.python
        self.image = args.image
        self.env: dict[str, Optional[str]] = {}
        if args.env_file:
            self.env.update(
                load_dotenv(Path(args.env_file).read_text(), environ=os.environ.copy())
            )
        for env_value in args.env or []:
            self.env.update(load_dotenv(env_value, environ=os.environ.copy()))
        self.secrets: dict[str, Optional[str]] = {}
        extended_environ = _get_extended_environ()
        if args.secrets_file:
            self.secrets.update(
                load_dotenv(
                    Path(args.secrets_file).read_text(), environ=extended_environ
                )
            )
        for secret in args.secrets or []:
            self.secrets.update(load_dotenv(secret, environ=extended_environ))
        self.flavor: Optional[SpaceHardware] = args.flavor
        self.timeout: Optional[str] = args.timeout
        self.detach: bool = args.detach
        self.namespace: Optional[str] = args.namespace
        self.token: Optional[str] = args.token
        self._repo = args.repo

    def run(self) -> None:
        """Execute UV command."""
        logging.set_verbosity(logging.INFO)
        api = HfApi(token=self.token)
        job = api.run_uv_job(
            script=self.script,
            script_args=self.script_args,
            dependencies=self.dependencies,
            python=self.python,
            image=self.image,
            env=self.env,
            secrets=self.secrets,
            flavor=self.flavor,
            timeout=self.timeout,
            namespace=self.namespace,
            _repo=self._repo,
        )

        # Always print the job ID to the user
        print(f"Job started with ID: {job.id}")
        print(f"View at: {job.url}")

        if self.detach:
            return

        # Now let's stream the logs
        for log in api.fetch_job_logs(job_id=job.id):
            print(log)


def _get_extended_environ() -> Dict[str, str]:
    extended_environ = os.environ.copy()
    if (token := get_token()) is not None:
        extended_environ["HF_TOKEN"] = token
    return extended_environ
