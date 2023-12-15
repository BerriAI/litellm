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
"""Contain helper class to retrieve/store token from/to local cache."""
import os
import warnings
from pathlib import Path
from typing import Optional

from .. import constants


class HfFolder:
    path_token = Path(constants.HF_TOKEN_PATH)
    # Private attribute. Will be removed in v0.15
    _old_path_token = Path(constants._OLD_HF_TOKEN_PATH)

    @classmethod
    def save_token(cls, token: str) -> None:
        """
        Save token, creating folder as needed.

        Token is saved in the huggingface home folder. You can configure it by setting
        the `HF_HOME` environment variable.

        Args:
            token (`str`):
                The token to save to the [`HfFolder`]
        """
        cls.path_token.parent.mkdir(parents=True, exist_ok=True)
        cls.path_token.write_text(token)

    @classmethod
    def get_token(cls) -> Optional[str]:
        """
        Get token or None if not existent.

        Note that a token can be also provided using the `HUGGING_FACE_HUB_TOKEN` environment variable.

        Token is saved in the huggingface home folder. You can configure it by setting
        the `HF_HOME` environment variable. Previous location was `~/.huggingface/token`.
        If token is found in old location but not in new location, it is copied there first.
        For more details, see https://github.com/huggingface/huggingface_hub/issues/1232.

        Returns:
            `str` or `None`: The token, `None` if it doesn't exist.
        """
        # 0. Check if token exist in old path but not new location
        try:
            cls._copy_to_new_path_and_warn()
        except Exception:  # if not possible (e.g. PermissionError), do not raise
            pass

        # 1. Is it set by environment variable ?
        token: Optional[str] = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token is not None:
            return token

        # 2. Is it set in token path ?
        try:
            return cls.path_token.read_text()
        except FileNotFoundError:
            return None

    @classmethod
    def delete_token(cls) -> None:
        """
        Deletes the token from storage. Does not fail if token does not exist.
        """
        try:
            cls.path_token.unlink()
        except FileNotFoundError:
            pass

        try:
            cls._old_path_token.unlink()
        except FileNotFoundError:
            pass

    @classmethod
    def _copy_to_new_path_and_warn(cls):
        if cls._old_path_token.exists() and not cls.path_token.exists():
            cls.save_token(cls._old_path_token.read_text())
            warnings.warn(
                f"A token has been found in `{cls._old_path_token}`. This is the old"
                " path where tokens were stored. The new location is"
                f" `{cls.path_token}` which is configurable using `HF_HOME` environment"
                " variable. Your token has been copied to this new location. You can"
                " now safely delete the old token file manually or use"
                " `huggingface-cli logout`."
            )
