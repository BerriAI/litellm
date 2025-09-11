# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Connection for the Google BigQuery DB-API."""

import weakref

from google.cloud import bigquery
from google.cloud.bigquery.dbapi import cursor
from google.cloud.bigquery.dbapi import _helpers


@_helpers.raise_on_closed("Operating on a closed connection.")
class Connection(object):
    """DB-API Connection to Google BigQuery.

    Args:
        client (Optional[google.cloud.bigquery.Client]):
            A REST API client used to connect to BigQuery. If not passed, a
            client is created using default options inferred from the environment.
        bqstorage_client(\
            Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient] \
        ):
            A client that uses the faster BigQuery Storage API to fetch rows from
            BigQuery. If not passed, it is created using the same credentials
            as ``client`` (provided that BigQuery Storage dependencies are installed).
        prefer_bqstorage_client (Optional[bool]):
            Prefer the BigQuery Storage client over the REST client. If Storage
            client isn't available, fall back to the REST client. Defaults to
            ``True``.
    """

    def __init__(
        self,
        client=None,
        bqstorage_client=None,
        prefer_bqstorage_client=True,
    ):
        if client is None:
            client = bigquery.Client()
            self._owns_client = True
        else:
            self._owns_client = False

        # A warning is already raised by the BQ Storage client factory factory if
        # instantiation fails, or if the given BQ Storage client instance is outdated.
        if not prefer_bqstorage_client:
            bqstorage_client = None
            self._owns_bqstorage_client = False
        elif bqstorage_client is None:
            bqstorage_client = client._ensure_bqstorage_client()
            self._owns_bqstorage_client = bqstorage_client is not None
        else:
            self._owns_bqstorage_client = False
            bqstorage_client = client._ensure_bqstorage_client(bqstorage_client)

        self._client = client
        self._bqstorage_client = bqstorage_client

        self._closed = False
        self._cursors_created = weakref.WeakSet()

    def close(self):
        """Close the connection and any cursors created from it.

        Any BigQuery clients explicitly passed to the constructor are *not*
        closed, only those created by the connection instance itself.
        """
        self._closed = True

        if self._owns_client:
            self._client.close()

        if self._owns_bqstorage_client:
            # There is no close() on the BQ Storage client itself.
            self._bqstorage_client._transport.grpc_channel.close()

        for cursor_ in self._cursors_created:
            if not cursor_._closed:
                cursor_.close()

    def commit(self):
        """No-op, but for consistency raise an error if connection is closed."""

    def cursor(self):
        """Return a new cursor object.

        Returns:
            google.cloud.bigquery.dbapi.Cursor: A DB-API cursor that uses this connection.
        """
        new_cursor = cursor.Cursor(self)
        self._cursors_created.add(new_cursor)
        return new_cursor


def connect(client=None, bqstorage_client=None, prefer_bqstorage_client=True):
    """Construct a DB-API connection to Google BigQuery.

    Args:
        client (Optional[google.cloud.bigquery.Client]):
            A REST API client used to connect to BigQuery. If not passed, a
            client is created using default options inferred from the environment.
        bqstorage_client(\
            Optional[google.cloud.bigquery_storage_v1.BigQueryReadClient] \
        ):
            A client that uses the faster BigQuery Storage API to fetch rows from
            BigQuery. If not passed, it is created using the same credentials
            as ``client`` (provided that BigQuery Storage dependencies are installed).
        prefer_bqstorage_client (Optional[bool]):
            Prefer the BigQuery Storage client over the REST client. If Storage
            client isn't available, fall back to the REST client. Defaults to
            ``True``.

    Returns:
        google.cloud.bigquery.dbapi.Connection: A new DB-API connection to BigQuery.
    """
    return Connection(client, bqstorage_client, prefer_bqstorage_client)
