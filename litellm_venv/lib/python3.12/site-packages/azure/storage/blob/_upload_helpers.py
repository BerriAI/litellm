# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from io import SEEK_SET, UnsupportedOperation
from typing import Any, cast, Dict, IO, Optional, TypeVar, TYPE_CHECKING

from azure.core.exceptions import (
    ResourceExistsError,
    ResourceModifiedError,
    HttpResponseError,
)

from ._encryption import (
    _ENCRYPTION_PROTOCOL_V1,
    _ENCRYPTION_PROTOCOL_V2,
    encrypt_blob,
    GCMBlobEncryptionStream,
    generate_blob_encryption_data,
    get_adjusted_upload_size,
    get_blob_encryptor_and_padder,
)
from ._generated.models import (
    AppendPositionAccessConditions,
    BlockLookupList,
    ModifiedAccessConditions,
)
from ._shared.models import StorageErrorCode
from ._shared.response_handlers import process_storage_error, return_response_headers
from ._shared.uploads import (
    AppendBlobChunkUploader,
    BlockBlobChunkUploader,
    PageBlobChunkUploader,
    upload_data_chunks,
    upload_substream_blocks,
)

if TYPE_CHECKING:
    from ._generated.operations import (
        AppendBlobOperations,
        BlockBlobOperations,
        PageBlobOperations,
    )
    from ._shared.models import StorageConfiguration

    BlobLeaseClient = TypeVar("BlobLeaseClient")

_LARGE_BLOB_UPLOAD_MAX_READ_BUFFER_SIZE = 4 * 1024 * 1024
_ERROR_VALUE_SHOULD_BE_SEEKABLE_STREAM = (
    "{0} should be a seekable file-like/io.IOBase type stream object."
)


def _convert_mod_error(error):
    message = error.message.replace(
        "The condition specified using HTTP conditional header(s) is not met.",
        "The specified blob already exists.",
    )
    message = message.replace("ConditionNotMet", "BlobAlreadyExists")
    overwrite_error = ResourceExistsError(
        message=message, response=error.response, error=error
    )
    overwrite_error.error_code = StorageErrorCode.blob_already_exists
    raise overwrite_error


def _any_conditions(
    modified_access_conditions=None, **kwargs
):  # pylint: disable=unused-argument
    return any(
        [
            modified_access_conditions.if_modified_since,
            modified_access_conditions.if_unmodified_since,
            modified_access_conditions.if_none_match,
            modified_access_conditions.if_match,
        ]
    )


def upload_block_blob(  # pylint: disable=too-many-locals, too-many-statements
    client: "BlockBlobOperations",
    stream: IO,
    overwrite: bool,
    encryption_options: Dict[str, Any],
    blob_settings: "StorageConfiguration",
    headers: Dict[str, Any],
    validate_content: bool,
    max_concurrency: Optional[int],
    length: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        if not overwrite and not _any_conditions(**kwargs):
            kwargs["modified_access_conditions"].if_none_match = "*"
        adjusted_count = length
        if (encryption_options.get("key") is not None) and (adjusted_count is not None):
            adjusted_count = get_adjusted_upload_size(
                adjusted_count, encryption_options["version"]
            )
        blob_headers = kwargs.pop("blob_headers", None)
        tier = kwargs.pop("standard_blob_tier", None)
        blob_tags_string = kwargs.pop("blob_tags_string", None)

        immutability_policy = kwargs.pop("immutability_policy", None)
        immutability_policy_expiry = (
            None if immutability_policy is None else immutability_policy.expiry_time
        )
        immutability_policy_mode = (
            None if immutability_policy is None else immutability_policy.policy_mode
        )
        legal_hold = kwargs.pop("legal_hold", None)
        progress_hook = kwargs.pop("progress_hook", None)

        # Do single put if the size is smaller than or equal config.max_single_put_size
        if adjusted_count is not None and (
            adjusted_count <= blob_settings.max_single_put_size
        ):
            data = stream.read(length or -1)
            if not isinstance(data, bytes):
                raise TypeError("Blob data should be of type bytes.")

            if encryption_options.get("key"):
                encryption_data, data = encrypt_blob(
                    data, encryption_options["key"], encryption_options["version"]
                )
                headers["x-ms-meta-encryptiondata"] = encryption_data

            response = client.upload(
                body=data,  # type: ignore [arg-type]
                content_length=adjusted_count,
                blob_http_headers=blob_headers,
                headers=headers,
                cls=return_response_headers,
                validate_content=validate_content,
                data_stream_total=adjusted_count,
                upload_stream_current=0,
                tier=tier.value if tier else None,
                blob_tags_string=blob_tags_string,
                immutability_policy_expiry=immutability_policy_expiry,
                immutability_policy_mode=immutability_policy_mode,
                legal_hold=legal_hold,
                **kwargs,
            )

            if progress_hook:
                progress_hook(adjusted_count, adjusted_count)

            return cast(Dict[str, Any], response)

        use_original_upload_path = (
            blob_settings.use_byte_buffer
            or validate_content
            or encryption_options.get("required")
            or blob_settings.max_block_size
            < blob_settings.min_large_block_upload_threshold
            or hasattr(stream, "seekable")
            and not stream.seekable()
            or not hasattr(stream, "seek")
            or not hasattr(stream, "tell")
        )

        if use_original_upload_path:
            total_size = length
            encryptor, padder = None, None
            if encryption_options and encryption_options.get("key"):
                cek, iv, encryption_metadata = generate_blob_encryption_data(
                    encryption_options["key"], encryption_options["version"]
                )
                headers["x-ms-meta-encryptiondata"] = encryption_metadata

                if encryption_options["version"] == _ENCRYPTION_PROTOCOL_V1:
                    encryptor, padder = get_blob_encryptor_and_padder(cek, iv, True)

                # Adjust total_size for encryption V2
                if encryption_options["version"] == _ENCRYPTION_PROTOCOL_V2:
                    # Adjust total_size for encryption V2
                    total_size = adjusted_count
                    # V2 wraps the data stream with an encryption stream
                    if cek is None:
                        raise ValueError(
                            "Generate encryption metadata failed. 'cek' is None."
                        )
                    stream = GCMBlobEncryptionStream(cek, stream)  # type: ignore [assignment]

            block_ids = upload_data_chunks(
                service=client,
                uploader_class=BlockBlobChunkUploader,
                total_size=total_size,
                chunk_size=blob_settings.max_block_size,
                max_concurrency=max_concurrency,
                stream=stream,
                validate_content=validate_content,
                progress_hook=progress_hook,
                encryptor=encryptor,
                padder=padder,
                headers=headers,
                **kwargs,
            )
        else:
            block_ids = upload_substream_blocks(
                service=client,
                uploader_class=BlockBlobChunkUploader,
                total_size=length,
                chunk_size=blob_settings.max_block_size,
                max_concurrency=max_concurrency,
                stream=stream,
                validate_content=validate_content,
                progress_hook=progress_hook,
                headers=headers,
                **kwargs,
            )

        block_lookup = BlockLookupList(committed=[], uncommitted=[], latest=[])
        block_lookup.latest = block_ids
        return cast(
            Dict[str, Any],
            client.commit_block_list(
                block_lookup,
                blob_http_headers=blob_headers,
                cls=return_response_headers,
                validate_content=validate_content,
                headers=headers,
                tier=tier.value if tier else None,
                blob_tags_string=blob_tags_string,
                immutability_policy_expiry=immutability_policy_expiry,
                immutability_policy_mode=immutability_policy_mode,
                legal_hold=legal_hold,
                **kwargs,
            ),
        )
    except HttpResponseError as error:
        try:
            process_storage_error(error)
        except ResourceModifiedError as mod_error:
            if not overwrite:
                _convert_mod_error(mod_error)
            raise


def upload_page_blob(
    client: "PageBlobOperations",
    overwrite: bool,
    encryption_options: Dict[str, Any],
    blob_settings: "StorageConfiguration",
    headers: Dict[str, Any],
    stream: IO,
    length: Optional[int] = None,
    validate_content: Optional[bool] = None,
    max_concurrency: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        if not overwrite and not _any_conditions(**kwargs):
            kwargs["modified_access_conditions"].if_none_match = "*"
        if length is None or length < 0:
            raise ValueError("A content length must be specified for a Page Blob.")
        if length % 512 != 0:
            raise ValueError(
                f"Invalid page blob size: {length}. "
                "The size must be aligned to a 512-byte boundary."
            )
        tier = None
        if kwargs.get("premium_page_blob_tier"):
            premium_page_blob_tier = kwargs.pop("premium_page_blob_tier")
            try:
                tier = premium_page_blob_tier.value
            except AttributeError:
                tier = premium_page_blob_tier

        if encryption_options and encryption_options.get("key"):
            cek, iv, encryption_data = generate_blob_encryption_data(
                encryption_options["key"], encryption_options["version"]
            )
            headers["x-ms-meta-encryptiondata"] = encryption_data

        blob_tags_string = kwargs.pop("blob_tags_string", None)
        progress_hook = kwargs.pop("progress_hook", None)

        response = cast(
            Dict[str, Any],
            client.create(
                content_length=0,
                blob_content_length=length,
                blob_sequence_number=None,  # type: ignore [arg-type]
                blob_http_headers=kwargs.pop("blob_headers", None),
                blob_tags_string=blob_tags_string,
                tier=tier,
                cls=return_response_headers,
                headers=headers,
                **kwargs,
            ),
        )
        if length == 0:
            return cast(Dict[str, Any], response)

        if encryption_options and encryption_options.get("key"):
            if encryption_options["version"] == _ENCRYPTION_PROTOCOL_V1:
                encryptor, padder = get_blob_encryptor_and_padder(cek, iv, False)
                kwargs["encryptor"] = encryptor
                kwargs["padder"] = padder

        kwargs["modified_access_conditions"] = ModifiedAccessConditions(
            if_match=response["etag"]
        )
        return cast(
            Dict[str, Any],
            upload_data_chunks(
                service=client,
                uploader_class=PageBlobChunkUploader,
                total_size=length,
                chunk_size=blob_settings.max_page_size,
                stream=stream,
                max_concurrency=max_concurrency,
                validate_content=validate_content,
                progress_hook=progress_hook,
                headers=headers,
                **kwargs,
            ),
        )

    except HttpResponseError as error:
        try:
            process_storage_error(error)
        except ResourceModifiedError as mod_error:
            if not overwrite:
                _convert_mod_error(mod_error)
            raise


def upload_append_blob(  # pylint: disable=unused-argument
    client: "AppendBlobOperations",
    overwrite: bool,
    encryption_options: Dict[str, Any],
    blob_settings: "StorageConfiguration",
    headers: Dict[str, Any],
    stream: IO,
    length: Optional[int] = None,
    validate_content: Optional[bool] = None,
    max_concurrency: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        if length == 0:
            return {}
        blob_headers = kwargs.pop("blob_headers", None)
        append_conditions = AppendPositionAccessConditions(
            max_size=kwargs.pop("maxsize_condition", None), append_position=None
        )
        blob_tags_string = kwargs.pop("blob_tags_string", None)
        progress_hook = kwargs.pop("progress_hook", None)

        try:
            if overwrite:
                client.create(
                    content_length=0,
                    blob_http_headers=blob_headers,
                    headers=headers,
                    blob_tags_string=blob_tags_string,
                    **kwargs,
                )
            return cast(
                Dict[str, Any],
                upload_data_chunks(
                    service=client,
                    uploader_class=AppendBlobChunkUploader,
                    total_size=length,
                    chunk_size=blob_settings.max_block_size,
                    stream=stream,
                    max_concurrency=max_concurrency,
                    validate_content=validate_content,
                    append_position_access_conditions=append_conditions,
                    progress_hook=progress_hook,
                    headers=headers,
                    **kwargs,
                ),
            )
        except HttpResponseError as error:
            if error.response.status_code != 404:  # type: ignore [union-attr]
                raise
            # rewind the request body if it is a stream
            if hasattr(stream, "read"):
                try:
                    # attempt to rewind the body to the initial position
                    stream.seek(0, SEEK_SET)
                except UnsupportedOperation as exc:
                    # if body is not seekable, then retry would not work
                    raise error from exc
            client.create(
                content_length=0,
                blob_http_headers=blob_headers,
                headers=headers,
                blob_tags_string=blob_tags_string,
                **kwargs,
            )
            return cast(
                Dict[str, Any],
                upload_data_chunks(
                    service=client,
                    uploader_class=AppendBlobChunkUploader,
                    total_size=length,
                    chunk_size=blob_settings.max_block_size,
                    stream=stream,
                    max_concurrency=max_concurrency,
                    validate_content=validate_content,
                    append_position_access_conditions=append_conditions,
                    progress_hook=progress_hook,
                    headers=headers,
                    **kwargs,
                ),
            )
    except HttpResponseError as error:
        process_storage_error(error)
