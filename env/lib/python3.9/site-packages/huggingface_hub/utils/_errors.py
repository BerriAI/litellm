from typing import Optional

from requests import HTTPError, Response

from ._fixes import JSONDecodeError


class HfHubHTTPError(HTTPError):
    """
    HTTPError to inherit from for any custom HTTP Error raised in HF Hub.

    Any HTTPError is converted at least into a `HfHubHTTPError`. If some information is
    sent back by the server, it will be added to the error message.

    Added details:
    - Request id from "X-Request-Id" header if exists.
    - Server error message from the header "X-Error-Message".
    - Server error message if we can found one in the response body.

    Example:
    ```py
        import requests
        from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

        response = get_session().post(...)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            print(str(e)) # formatted message
            e.request_id, e.server_message # details returned by server

            # Complete the error message with additional information once it's raised
            e.append_to_message("\n`create_commit` expects the repository to exist.")
            raise
    ```
    """

    request_id: Optional[str] = None
    server_message: Optional[str] = None

    def __init__(self, message: str, response: Optional[Response] = None):
        # Parse server information if any.
        if response is not None:
            self.request_id = response.headers.get("X-Request-Id")
            try:
                server_data = response.json()
            except JSONDecodeError:
                server_data = {}

            # Retrieve server error message from multiple sources
            server_message_from_headers = response.headers.get("X-Error-Message")
            server_message_from_body = server_data.get("error")
            server_multiple_messages_from_body = "\n".join(
                error["message"] for error in server_data.get("errors", []) if "message" in error
            )

            # Concatenate error messages
            _server_message = ""
            if server_message_from_headers is not None:  # from headers
                _server_message += server_message_from_headers + "\n"
            if server_message_from_body is not None:  # from body "error"
                if isinstance(server_message_from_body, list):
                    server_message_from_body = "\n".join(server_message_from_body)
                if server_message_from_body not in _server_message:
                    _server_message += server_message_from_body + "\n"
            if server_multiple_messages_from_body is not None:  # from body "errors"
                if server_multiple_messages_from_body not in _server_message:
                    _server_message += server_multiple_messages_from_body + "\n"
            _server_message = _server_message.strip()

            # Set message to `HfHubHTTPError` (if any)
            if _server_message != "":
                self.server_message = _server_message

        super().__init__(
            _format_error_message(
                message,
                request_id=self.request_id,
                server_message=self.server_message,
            ),
            response=response,
        )

    def append_to_message(self, additional_message: str) -> None:
        """Append additional information to the `HfHubHTTPError` initial message."""
        self.args = (self.args[0] + additional_message,) + self.args[1:]


class RepositoryNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or
    with a private repo name the user does not have access to.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<non_existent_repository>")
    (...)
    huggingface_hub.utils._errors.RepositoryNotFoundError: 401 Client Error. (Request ID: PvMw_VjBMjVdMz53WKIzP)

    Repository Not Found for url: https://huggingface.co/api/models/%3Cnon_existent_repository%3E.
    Please make sure you specified the correct `repo_id` and `repo_type`.
    If the repo is private, make sure you are authenticated.
    Invalid username or password.
    ```
    """


class GatedRepoError(RepositoryNotFoundError):
    """
    Raised when trying to access a gated repository for which the user is not on the
    authorized list.

    Note: derives from `RepositoryNotFoundError` to ensure backward compatibility.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<gated_repository>")
    (...)
    huggingface_hub.utils._errors.GatedRepoError: 403 Client Error. (Request ID: ViT1Bf7O_026LGSQuVqfa)

    Cannot access gated repo for url https://huggingface.co/api/models/ardent-figment/gated-model.
    Access to model ardent-figment/gated-model is restricted and you are not in the authorized list.
    Visit https://huggingface.co/ardent-figment/gated-model to ask for access.
    ```
    """


class RevisionNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository but an invalid
    revision.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', 'config.json', revision='<non-existent-revision>')
    (...)
    huggingface_hub.utils._errors.RevisionNotFoundError: 404 Client Error. (Request ID: Mwhe_c3Kt650GcdKEFomX)

    Revision Not Found for url: https://huggingface.co/bert-base-cased/resolve/%3Cnon-existent-revision%3E/config.json.
    ```
    """


class EntryNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository and revision
    but an invalid filename.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-existent-file>')
    (...)
    huggingface_hub.utils._errors.EntryNotFoundError: 404 Client Error. (Request ID: 53pNl6M0MxsnG5Sw8JA6x)

    Entry Not Found for url: https://huggingface.co/bert-base-cased/resolve/main/%3Cnon-existent-file%3E.
    ```
    """


class LocalEntryNotFoundError(EntryNotFoundError, FileNotFoundError, ValueError):
    """
    Raised when trying to access a file that is not on the disk when network is
    disabled or unavailable (connection issue). The entry may exist on the Hub.

    Note: `ValueError` type is to ensure backward compatibility.
    Note: `LocalEntryNotFoundError` derives from `HTTPError` because of `EntryNotFoundError`
          even when it is not a network issue.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-cached-file>',  local_files_only=True)
    (...)
    huggingface_hub.utils._errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.
    ```
    """

    def __init__(self, message: str):
        super().__init__(message, response=None)


class BadRequestError(HfHubHTTPError, ValueError):
    """
    Raised by `hf_raise_for_status` when the server returns a HTTP 400 error.

    Example:

    ```py
    >>> resp = requests.post("hf.co/api/check", ...)
    >>> hf_raise_for_status(resp, endpoint_name="check")
    huggingface_hub.utils._errors.BadRequestError: Bad request for check endpoint: {details} (Request ID: XXX)
    ```
    """


def hf_raise_for_status(response: Response, endpoint_name: Optional[str] = None) -> None:
    """
    Internal version of `response.raise_for_status()` that will refine a
    potential HTTPError. Raised exception will be an instance of `HfHubHTTPError`.

    This helper is meant to be the unique method to raise_for_status when making a call
    to the Hugging Face Hub.

    Example:
    ```py
        import requests
        from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

        response = get_session().post(...)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            print(str(e)) # formatted message
            e.request_id, e.server_message # details returned by server

            # Complete the error message with additional information once it's raised
            e.append_to_message("\n`create_commit` expects the repository to exist.")
            raise
    ```

    Args:
        response (`Response`):
            Response from the server.
        endpoint_name (`str`, *optional*):
            Name of the endpoint that has been called. If provided, the error message
            will be more complete.

    <Tip warning={true}>

    Raises when the request has failed:

        - [`~utils.RepositoryNotFoundError`]
            If the repository to download from cannot be found. This may be because it
            doesn't exist, because `repo_type` is not set correctly, or because the repo
            is `private` and you do not have access.
        - [`~utils.GatedRepoError`]
            If the repository exists but is gated and the user is not on the authorized
            list.
        - [`~utils.RevisionNotFoundError`]
            If the repository exists but the revision couldn't be find.
        - [`~utils.EntryNotFoundError`]
            If the repository exists but the entry (e.g. the requested file) couldn't be
            find.
        - [`~utils.BadRequestError`]
            If request failed with a HTTP 400 BadRequest error.
        - [`~utils.HfHubHTTPError`]
            If request failed for a reason not listed above.

    </Tip>
    """
    try:
        response.raise_for_status()
    except HTTPError as e:
        error_code = response.headers.get("X-Error-Code")

        if error_code == "RevisionNotFound":
            message = f"{response.status_code} Client Error." + "\n\n" + f"Revision Not Found for url: {response.url}."
            raise RevisionNotFoundError(message, response) from e

        elif error_code == "EntryNotFound":
            message = f"{response.status_code} Client Error." + "\n\n" + f"Entry Not Found for url: {response.url}."
            raise EntryNotFoundError(message, response) from e

        elif error_code == "GatedRepo":
            message = (
                f"{response.status_code} Client Error." + "\n\n" + f"Cannot access gated repo for url {response.url}."
            )
            raise GatedRepoError(message, response) from e

        elif error_code == "RepoNotFound" or response.status_code == 401:
            # 401 is misleading as it is returned for:
            #    - private and gated repos if user is not authenticated
            #    - missing repos
            # => for now, we process them as `RepoNotFound` anyway.
            # See https://gist.github.com/Wauplin/46c27ad266b15998ce56a6603796f0b9
            message = (
                f"{response.status_code} Client Error."
                + "\n\n"
                + f"Repository Not Found for url: {response.url}."
                + "\nPlease make sure you specified the correct `repo_id` and"
                " `repo_type`.\nIf you are trying to access a private or gated repo,"
                " make sure you are authenticated."
            )
            raise RepositoryNotFoundError(message, response) from e

        elif response.status_code == 400:
            message = (
                f"\n\nBad request for {endpoint_name} endpoint:" if endpoint_name is not None else "\n\nBad request:"
            )
            raise BadRequestError(message, response=response) from e

        # Convert `HTTPError` into a `HfHubHTTPError` to display request information
        # as well (request id and/or server error message)
        raise HfHubHTTPError(str(e), response=response) from e


def _format_error_message(message: str, request_id: Optional[str], server_message: Optional[str]) -> str:
    """
    Format the `HfHubHTTPError` error message based on initial message and information
    returned by the server.

    Used when initializing `HfHubHTTPError`.
    """
    # Add message from response body
    if server_message is not None and len(server_message) > 0 and server_message.lower() not in message.lower():
        if "\n\n" in message:
            message += "\n" + server_message
        else:
            message += "\n\n" + server_message

    # Add Request ID
    if request_id is not None and str(request_id).lower() not in message.lower():
        request_id_message = f" (Request ID: {request_id})"
        if "\n" in message:
            newline_index = message.index("\n")
            message = message[:newline_index] + request_id_message + message[newline_index:]
        else:
            message += request_id_message

    return message
