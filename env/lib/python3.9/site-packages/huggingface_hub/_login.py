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
"""Contains methods to login to the Hub."""
import os
import subprocess
from functools import partial
from getpass import getpass
from typing import Optional

from .commands._cli_utils import ANSI
from .commands.delete_cache import _ask_for_confirmation_no_tui
from .hf_api import get_token_permission
from .utils import (
    HfFolder,
    capture_output,
    is_google_colab,
    is_notebook,
    list_credential_helpers,
    logging,
    run_subprocess,
    set_git_credential,
    unset_git_credential,
)


logger = logging.get_logger(__name__)


def login(
    token: Optional[str] = None,
    add_to_git_credential: bool = False,
    new_session: bool = True,
    write_permission: bool = False,
) -> None:
    """Login the machine to access the Hub.

    The `token` is persisted in cache and set as a git credential. Once done, the machine
    is logged in and the access token will be available across all `huggingface_hub`
    components. If `token` is not provided, it will be prompted to the user either with
    a widget (in a notebook) or via the terminal.

    To login from outside of a script, one can also use `huggingface-cli login` which is
    a cli command that wraps [`login`].

    <Tip>
    [`login`] is a drop-in replacement method for [`notebook_login`] as it wraps and
    extends its capabilities.
    </Tip>

    <Tip>
    When the token is not passed, [`login`] will automatically detect if the script runs
    in a notebook or not. However, this detection might not be accurate due to the
    variety of notebooks that exists nowadays. If that is the case, you can always force
    the UI by using [`notebook_login`] or [`interpreter_login`].
    </Tip>

    Args:
        token (`str`, *optional*):
            User access token to generate from https://huggingface.co/settings/token.
        add_to_git_credential (`bool`, defaults to `False`):
            If `True`, token will be set as git credential. If no git credential helper
            is configured, a warning will be displayed to the user. If `token` is `None`,
            the value of `add_to_git_credential` is ignored and will be prompted again
            to the end user.
        new_session (`bool`, defaults to `True`):
            If `True`, will request a token even if one is already saved on the machine.
        write_permission (`bool`, defaults to `False`):
            If `True`, requires a token with write permission.
    Raises:
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If an organization token is passed. Only personal account tokens are valid
            to login.
        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If token is invalid.
        [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError)
            If running in a notebook but `ipywidgets` is not installed.
    """
    if token is not None:
        if not add_to_git_credential:
            print(
                "Token will not been saved to git credential helper. Pass"
                " `add_to_git_credential=True` if you want to set the git"
                " credential as well."
            )
        _login(token, add_to_git_credential=add_to_git_credential, write_permission=write_permission)
    elif is_notebook():
        notebook_login(new_session=new_session, write_permission=write_permission)
    else:
        interpreter_login(new_session=new_session, write_permission=write_permission)


def logout() -> None:
    """Logout the machine from the Hub.

    Token is deleted from the machine and removed from git credential.
    """
    token = HfFolder.get_token()
    if token is None:
        print("Not logged in!")
        return
    HfFolder.delete_token()
    unset_git_credential()
    print("Successfully logged out.")


###
# Interpreter-based login (text)
###


def interpreter_login(new_session: bool = True, write_permission: bool = False) -> None:
    """
    Displays a prompt to login to the HF website and store the token.

    This is equivalent to [`login`] without passing a token when not run in a notebook.
    [`interpreter_login`] is useful if you want to force the use of the terminal prompt
    instead of a notebook widget.

    For more details, see [`login`].

    Args:
        new_session (`bool`, defaults to `True`):
            If `True`, will request a token even if one is already saved on the machine.
        write_permission (`bool`, defaults to `False`):
            If `True`, requires a token with write permission.

    """
    if not new_session and _current_token_okay(write_permission=write_permission):
        print("User is already logged in.")
        return

    print("""
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
    """)
    if HfFolder.get_token() is not None:
        print(
            "    A token is already saved on your machine. Run `huggingface-cli"
            " whoami` to get more information or `huggingface-cli logout` if you want"
            " to log out."
        )
        print("    Setting a new token will erase the existing one.")

    print("    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .")
    if os.name == "nt":
        print("Token can be pasted using 'Right-Click'.")
    token = getpass("Token: ")
    add_to_git_credential = _ask_for_confirmation_no_tui("Add token as git credential?")

    _login(token=token, add_to_git_credential=add_to_git_credential, write_permission=write_permission)


###
# Notebook-based login (widget)
###

NOTEBOOK_LOGIN_PASSWORD_HTML = """<center> <img
src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg
alt='Hugging Face'> <br> Immediately click login after typing your password or
it might be stored in plain text in this notebook file. </center>"""


NOTEBOOK_LOGIN_TOKEN_HTML_START = """<center> <img
src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg
alt='Hugging Face'> <br> Copy a token from <a
href="https://huggingface.co/settings/tokens" target="_blank">your Hugging Face
tokens page</a> and paste it below. <br> Immediately click login after copying
your token or it might be stored in plain text in this notebook file. </center>"""


NOTEBOOK_LOGIN_TOKEN_HTML_END = """
<b>Pro Tip:</b> If you don't already have one, you can create a dedicated
'notebooks' token with 'write' access, that you can then easily reuse for all
notebooks. </center>"""


def notebook_login(new_session: bool = True, write_permission: bool = False) -> None:
    """
    Displays a widget to login to the HF website and store the token.

    This is equivalent to [`login`] without passing a token when run in a notebook.
    [`notebook_login`] is useful if you want to force the use of the notebook widget
    instead of a prompt in the terminal.

    For more details, see [`login`].

    Args:
        new_session (`bool`, defaults to `True`):
            If `True`, will request a token even if one is already saved on the machine.
        write_permission (`bool`, defaults to `False`):
            If `True`, requires a token with write permission.
    """
    try:
        import ipywidgets.widgets as widgets  # type: ignore
        from IPython.display import display  # type: ignore
    except ImportError:
        raise ImportError(
            "The `notebook_login` function can only be used in a notebook (Jupyter or"
            " Colab) and you need the `ipywidgets` module: `pip install ipywidgets`."
        )
    if not new_session and _current_token_okay(write_permission=write_permission):
        print("User is already logged in.")
        return

    box_layout = widgets.Layout(display="flex", flex_flow="column", align_items="center", width="50%")

    token_widget = widgets.Password(description="Token:")
    git_checkbox_widget = widgets.Checkbox(value=True, description="Add token as git credential?")
    token_finish_button = widgets.Button(description="Login")

    login_token_widget = widgets.VBox(
        [
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_START),
            token_widget,
            git_checkbox_widget,
            token_finish_button,
            widgets.HTML(NOTEBOOK_LOGIN_TOKEN_HTML_END),
        ],
        layout=box_layout,
    )
    display(login_token_widget)

    # On click events
    def login_token_event(t, write_permission: bool = False):
        """
        Event handler for the login button.

        Args:
            write_permission (`bool`, defaults to `False`):
                If `True`, requires a token with write permission.
        """
        token = token_widget.value
        add_to_git_credential = git_checkbox_widget.value
        # Erase token and clear value to make sure it's not saved in the notebook.
        token_widget.value = ""
        # Hide inputs
        login_token_widget.children = [widgets.Label("Connecting...")]
        try:
            with capture_output() as captured:
                _login(token, add_to_git_credential=add_to_git_credential, write_permission=write_permission)
            message = captured.getvalue()
        except Exception as error:
            message = str(error)
        # Print result (success message or error)
        login_token_widget.children = [widgets.Label(line) for line in message.split("\n") if line.strip()]

    token_finish_button.on_click(partial(login_token_event, write_permission=write_permission))


###
# Login private helpers
###


def _login(token: str, add_to_git_credential: bool, write_permission: bool = False) -> None:
    if token.startswith("api_org"):
        raise ValueError("You must use your personal account token, not an organization token.")

    permission = get_token_permission(token)
    if permission is None:
        raise ValueError("Invalid token passed!")
    elif write_permission and permission != "write":
        raise ValueError(
            "Token is valid but is 'read-only' and a 'write' token is required.\nPlease provide a new token with"
            " correct permission."
        )
    print(f"Token is valid (permission: {permission}).")

    if add_to_git_credential:
        if _is_git_credential_helper_configured():
            set_git_credential(token)
            print(
                "Your token has been saved in your configured git credential helpers"
                + f" ({','.join(list_credential_helpers())})."
            )
        else:
            print("Token has not been saved to git credential helper.")

    HfFolder.save_token(token)
    print(f"Your token has been saved to {HfFolder.path_token}")
    print("Login successful")


def _current_token_okay(write_permission: bool = False):
    """Check if the current token is valid.

    Args:
        write_permission (`bool`, defaults to `False`):
            If `True`, requires a token with write permission.

    Returns:
        `bool`: `True` if the current token is valid, `False` otherwise.
    """
    permission = get_token_permission()
    if permission is None or (write_permission and permission != "write"):
        return False
    return True


def _is_git_credential_helper_configured() -> bool:
    """Check if a git credential helper is configured.

    Warns user if not the case (except for Google Colab where "store" is set by default
    by `huggingface_hub`).
    """
    helpers = list_credential_helpers()
    if len(helpers) > 0:
        return True  # Do not warn: at least 1 helper is set

    # Only in Google Colab to avoid the warning message
    # See https://github.com/huggingface/huggingface_hub/issues/1043#issuecomment-1247010710
    if is_google_colab():
        _set_store_as_git_credential_helper_globally()
        return True  # Do not warn: "store" is used by default in Google Colab

    # Otherwise, warn user
    print(
        ANSI.red(
            "Cannot authenticate through git-credential as no helper is defined on your"
            " machine.\nYou might have to re-authenticate when pushing to the Hugging"
            " Face Hub.\nRun the following command in your terminal in case you want to"
            " set the 'store' credential helper as default.\n\ngit config --global"
            " credential.helper store\n\nRead"
            " https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more"
            " details."
        )
    )
    return False


def _set_store_as_git_credential_helper_globally() -> None:
    """Set globally the credential.helper to `store`.

    To be used only in Google Colab as we assume the user doesn't care about the git
    credential config. It is the only particular case where we don't want to display the
    warning message in [`notebook_login()`].

    Related:
    - https://github.com/huggingface/huggingface_hub/issues/1043
    - https://github.com/huggingface/huggingface_hub/issues/1051
    - https://git-scm.com/docs/git-credential-store
    """
    try:
        run_subprocess("git config --global credential.helper store")
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)
