# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. team.
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
"""Contains the 'hf cache' command group with 'scan' and 'delete' subcommands."""

import os
import time
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Literal, Optional, Union

from ..utils import (
    CachedRepoInfo,
    CachedRevisionInfo,
    CacheNotFound,
    HFCacheInfo,
    scan_cache_dir,
)
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI, tabulate


# --- DELETE helpers (from delete_cache.py) ---
try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    from InquirerPy.separator import Separator

    _inquirer_py_available = True
except ImportError:
    _inquirer_py_available = False

SortingOption_T = Literal["alphabetical", "lastUpdated", "lastUsed", "size"]
_CANCEL_DELETION_STR = "CANCEL_DELETION"


def require_inquirer_py(fn: Callable) -> Callable:
    @wraps(fn)
    def _inner(*args, **kwargs):
        if not _inquirer_py_available:
            raise ImportError(
                "The 'cache delete' command requires extra dependencies for the TUI.\n"
                "Please run 'pip install huggingface_hub[cli]' to install them.\n"
                "Otherwise, disable TUI using the '--disable-tui' flag."
            )
        return fn(*args, **kwargs)

    return _inner


class CacheCommand(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        cache_parser = parser.add_parser("cache", help="Manage local cache directory.")
        cache_subparsers = cache_parser.add_subparsers(
            dest="cache_command", help="Cache subcommands"
        )

        # Show help if no subcommand is provided
        cache_parser.set_defaults(func=lambda args: cache_parser.print_help())

        # Scan subcommand
        scan_parser = cache_subparsers.add_parser("scan", help="Scan cache directory.")
        scan_parser.add_argument(
            "--dir",
            type=str,
            default=None,
            help="cache directory to scan (optional). Default to the default HuggingFace cache.",
        )
        scan_parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="show a more verbose output",
        )
        scan_parser.set_defaults(func=CacheCommand, cache_command="scan")
        # Delete subcommand
        delete_parser = cache_subparsers.add_parser(
            "delete", help="Delete revisions from the cache directory."
        )
        delete_parser.add_argument(
            "--dir",
            type=str,
            default=None,
            help="cache directory (optional). Default to the default HuggingFace cache.",
        )
        delete_parser.add_argument(
            "--disable-tui",
            action="store_true",
            help=(
                "Disable Terminal User Interface (TUI) mode. Useful if your platform/terminal doesn't support the multiselect menu."
            ),
        )
        delete_parser.add_argument(
            "--sort",
            nargs="?",
            choices=["alphabetical", "lastUpdated", "lastUsed", "size"],
            help=(
                "Sort repositories by the specified criteria. Options: "
                "'alphabetical' (A-Z), "
                "'lastUpdated' (newest first), "
                "'lastUsed' (most recent first), "
                "'size' (largest first)."
            ),
        )
        delete_parser.set_defaults(func=CacheCommand, cache_command="delete")

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.verbosity: int = getattr(args, "verbose", 0)
        self.cache_dir: Optional[str] = getattr(args, "dir", None)
        self.disable_tui: bool = getattr(args, "disable_tui", False)
        self.sort_by: Optional[SortingOption_T] = getattr(args, "sort", None)
        self.cache_command: Optional[str] = getattr(args, "cache_command", None)

    def run(self):
        if self.cache_command == "scan":
            self._run_scan()
        elif self.cache_command == "delete":
            self._run_delete()
        else:
            print(
                "Please specify a cache subcommand (scan or delete). Use -h for help."
            )

    def _run_scan(self):
        try:
            t0 = time.time()
            hf_cache_info = scan_cache_dir(self.cache_dir)
            t1 = time.time()
        except CacheNotFound as exc:
            cache_dir = exc.cache_dir
            print(f"Cache directory not found: {cache_dir}")
            return
        print(get_table(hf_cache_info, verbosity=self.verbosity))
        print(
            f"\nDone in {round(t1 - t0, 1)}s. Scanned {len(hf_cache_info.repos)} repo(s)"
            f" for a total of {ANSI.red(hf_cache_info.size_on_disk_str)}."
        )
        if len(hf_cache_info.warnings) > 0:
            message = f"Got {len(hf_cache_info.warnings)} warning(s) while scanning."
            if self.verbosity >= 3:
                print(ANSI.gray(message))
                for warning in hf_cache_info.warnings:
                    print(ANSI.gray(warning))
            else:
                print(ANSI.gray(message + " Use -vvv to print details."))

    def _run_delete(self):
        hf_cache_info = scan_cache_dir(self.cache_dir)
        if self.disable_tui:
            selected_hashes = _manual_review_no_tui(
                hf_cache_info, preselected=[], sort_by=self.sort_by
            )
        else:
            selected_hashes = _manual_review_tui(
                hf_cache_info, preselected=[], sort_by=self.sort_by
            )
        if len(selected_hashes) > 0 and _CANCEL_DELETION_STR not in selected_hashes:
            confirm_message = (
                _get_expectations_str(hf_cache_info, selected_hashes)
                + " Confirm deletion ?"
            )
            if self.disable_tui:
                confirmed = _ask_for_confirmation_no_tui(confirm_message)
            else:
                confirmed = _ask_for_confirmation_tui(confirm_message)
            if confirmed:
                strategy = hf_cache_info.delete_revisions(*selected_hashes)
                print("Start deletion.")
                strategy.execute()
                print(
                    f"Done. Deleted {len(strategy.repos)} repo(s) and"
                    f" {len(strategy.snapshots)} revision(s) for a total of"
                    f" {strategy.expected_freed_size_str}."
                )
                return
        print("Deletion is cancelled. Do nothing.")


def get_table(hf_cache_info: HFCacheInfo, *, verbosity: int = 0) -> str:
    if verbosity == 0:
        return tabulate(
            rows=[
                [
                    repo.repo_id,
                    repo.repo_type,
                    "{:>12}".format(repo.size_on_disk_str),
                    repo.nb_files,
                    repo.last_accessed_str,
                    repo.last_modified_str,
                    ", ".join(sorted(repo.refs)),
                    str(repo.repo_path),
                ]
                for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path)
            ],
            headers=[
                "REPO ID",
                "REPO TYPE",
                "SIZE ON DISK",
                "NB FILES",
                "LAST_ACCESSED",
                "LAST_MODIFIED",
                "REFS",
                "LOCAL PATH",
            ],
        )
    else:
        return tabulate(
            rows=[
                [
                    repo.repo_id,
                    repo.repo_type,
                    revision.commit_hash,
                    "{:>12}".format(revision.size_on_disk_str),
                    revision.nb_files,
                    revision.last_modified_str,
                    ", ".join(sorted(revision.refs)),
                    str(revision.snapshot_path),
                ]
                for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path)
                for revision in sorted(
                    repo.revisions, key=lambda revision: revision.commit_hash
                )
            ],
            headers=[
                "REPO ID",
                "REPO TYPE",
                "REVISION",
                "SIZE ON DISK",
                "NB FILES",
                "LAST_MODIFIED",
                "REFS",
                "LOCAL PATH",
            ],
        )


def _get_repo_sorting_key(
    repo: CachedRepoInfo, sort_by: Optional[SortingOption_T] = None
):
    if sort_by == "alphabetical":
        return (repo.repo_type, repo.repo_id.lower())
    elif sort_by == "lastUpdated":
        return -max(rev.last_modified for rev in repo.revisions)
    elif sort_by == "lastUsed":
        return -repo.last_accessed
    elif sort_by == "size":
        return -repo.size_on_disk
    else:
        return (repo.repo_type, repo.repo_id)


@require_inquirer_py
def _manual_review_tui(
    hf_cache_info: HFCacheInfo,
    preselected: List[str],
    sort_by: Optional[SortingOption_T] = None,
) -> List[str]:
    choices = _get_tui_choices_from_scan(
        repos=hf_cache_info.repos, preselected=preselected, sort_by=sort_by
    )
    checkbox = inquirer.checkbox(
        message="Select revisions to delete:",
        choices=choices,
        cycle=False,
        height=100,
        instruction=_get_expectations_str(
            hf_cache_info,
            selected_hashes=[
                c.value for c in choices if isinstance(c, Choice) and c.enabled
            ],
        ),
        long_instruction="Press <space> to select, <enter> to validate and <ctrl+c> to quit without modification.",
        transformer=lambda result: f"{len(result)} revision(s) selected.",
    )

    def _update_expectations(_):
        checkbox._instruction = _get_expectations_str(
            hf_cache_info,
            selected_hashes=[
                choice["value"]
                for choice in checkbox.content_control.choices
                if choice["enabled"]
            ],
        )

    checkbox.kb_func_lookup["toggle"].append({"func": _update_expectations})
    try:
        return checkbox.execute()
    except KeyboardInterrupt:
        return []


@require_inquirer_py
def _ask_for_confirmation_tui(message: str, default: bool = True) -> bool:
    return inquirer.confirm(message, default=default).execute()


def _get_tui_choices_from_scan(
    repos: Iterable[CachedRepoInfo],
    preselected: List[str],
    sort_by: Optional[SortingOption_T] = None,
) -> List:
    choices: List[Union["Choice", "Separator"]] = []
    choices.append(
        Choice(
            _CANCEL_DELETION_STR,
            name="None of the following (if selected, nothing will be deleted).",
            enabled=False,
        )
    )
    sorted_repos = sorted(repos, key=lambda repo: _get_repo_sorting_key(repo, sort_by))
    for repo in sorted_repos:
        choices.append(
            Separator(
                f"\n{repo.repo_type.capitalize()} {repo.repo_id} ({repo.size_on_disk_str}, used {repo.last_accessed_str})"
            )
        )
        for revision in sorted(repo.revisions, key=_revision_sorting_order):
            choices.append(
                Choice(
                    revision.commit_hash,
                    name=(
                        f"{revision.commit_hash[:8]}: {', '.join(sorted(revision.refs)) or '(detached)'} # modified {revision.last_modified_str}"
                    ),
                    enabled=revision.commit_hash in preselected,
                )
            )
    return choices


def _manual_review_no_tui(
    hf_cache_info: HFCacheInfo,
    preselected: List[str],
    sort_by: Optional[SortingOption_T] = None,
) -> List[str]:
    fd, tmp_path = mkstemp(suffix=".txt")
    os.close(fd)
    lines = []
    sorted_repos = sorted(
        hf_cache_info.repos, key=lambda repo: _get_repo_sorting_key(repo, sort_by)
    )
    for repo in sorted_repos:
        lines.append(
            f"\n# {repo.repo_type.capitalize()} {repo.repo_id} ({repo.size_on_disk_str}, used {repo.last_accessed_str})"
        )
        for revision in sorted(repo.revisions, key=_revision_sorting_order):
            lines.append(
                f"{'' if revision.commit_hash in preselected else '#'}   {revision.commit_hash} # Refs: {', '.join(sorted(revision.refs)) or '(detached)'} # modified {revision.last_modified_str}"
            )
    with open(tmp_path, "w") as f:
        f.write(_MANUAL_REVIEW_NO_TUI_INSTRUCTIONS)
        f.write("\n".join(lines))
    instructions = f"""
    TUI is disabled. In order to select which revisions you want to delete, please edit
    the following file using the text editor of your choice. Instructions for manual
    editing are located at the beginning of the file. Edit the file, save it and confirm
    to continue.
    File to edit: {ANSI.bold(tmp_path)}
    """
    print("\n".join(line.strip() for line in instructions.strip().split("\n")))
    while True:
        selected_hashes = _read_manual_review_tmp_file(tmp_path)
        if _ask_for_confirmation_no_tui(
            _get_expectations_str(hf_cache_info, selected_hashes) + " Continue ?",
            default=False,
        ):
            break
    os.remove(tmp_path)
    return sorted(selected_hashes)


def _ask_for_confirmation_no_tui(message: str, default: bool = True) -> bool:
    YES = ("y", "yes", "1")
    NO = ("n", "no", "0")
    DEFAULT = ""
    ALL = YES + NO + (DEFAULT,)
    full_message = message + (" (Y/n) " if default else " (y/N) ")
    while True:
        answer = input(full_message).lower()
        if answer == DEFAULT:
            return default
        if answer in YES:
            return True
        if answer in NO:
            return False
        print(f"Invalid input. Must be one of {ALL}")


def _get_expectations_str(
    hf_cache_info: HFCacheInfo, selected_hashes: List[str]
) -> str:
    if _CANCEL_DELETION_STR in selected_hashes:
        return "Nothing will be deleted."
    strategy = hf_cache_info.delete_revisions(*selected_hashes)
    return f"{len(selected_hashes)} revisions selected counting for {strategy.expected_freed_size_str}."


def _read_manual_review_tmp_file(tmp_path: str) -> List[str]:
    with open(tmp_path) as f:
        content = f.read()
    lines = [line.strip() for line in content.split("\n")]
    selected_lines = [line for line in lines if not line.startswith("#")]
    selected_hashes = [line.split("#")[0].strip() for line in selected_lines]
    return [hash for hash in selected_hashes if len(hash) > 0]


_MANUAL_REVIEW_NO_TUI_INSTRUCTIONS = f"""
# INSTRUCTIONS
# ------------
# This is a temporary file created by running `hf cache delete --disable-tui`. It contains a set of revisions that can be deleted from your local cache directory.
#
# Please manually review the revisions you want to delete:
#   - Revision hashes can be commented out with '#'.
#   - Only non-commented revisions in this file will be deleted.
#   - Revision hashes that are removed from this file are ignored as well.
#   - If `{_CANCEL_DELETION_STR}` line is uncommented, the all cache deletion is cancelled and no changes will be applied.
#
# Once you've manually reviewed this file, please confirm deletion in the terminal. This file will be automatically removed once done.
# ------------

# KILL SWITCH
# ------------
# Un-comment following line to completely cancel the deletion process
# {_CANCEL_DELETION_STR}
# ------------

# REVISIONS
# ------------
""".strip()


def _revision_sorting_order(revision: CachedRevisionInfo) -> Any:
    return revision.last_modified
