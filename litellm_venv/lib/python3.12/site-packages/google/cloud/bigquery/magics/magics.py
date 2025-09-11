# Copyright 2018 Google LLC
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

"""IPython Magics

Install ``bigquery-magics`` and call ``%load_ext bigquery_magics`` to use the
``%%bigquery`` cell magic.

See the `BigQuery Magics reference documentation
<https://googleapis.dev/python/bigquery-magics/latest/>`_.
"""

from __future__ import print_function

import re
import ast
import copy
import functools
import sys
import time
import warnings
from concurrent import futures

try:
    import IPython  # type: ignore
    from IPython import display  # type: ignore
    from IPython.core import magic_arguments  # type: ignore
except ImportError:
    raise ImportError("This module can only be loaded in IPython.")

from google.api_core import client_info
from google.api_core import client_options
from google.api_core.exceptions import NotFound
import google.auth  # type: ignore
from google.cloud import bigquery
import google.cloud.bigquery.dataset
from google.cloud.bigquery import _versions_helpers
from google.cloud.bigquery import exceptions
from google.cloud.bigquery.dbapi import _helpers
from google.cloud.bigquery.magics import line_arg_parser as lap

try:
    import bigquery_magics  # type: ignore
except ImportError:
    bigquery_magics = None

IPYTHON_USER_AGENT = "ipython-{}".format(IPython.__version__)  # type: ignore


class Context(object):
    """Storage for objects to be used throughout an IPython notebook session.

    A Context object is initialized when the ``magics`` module is imported,
    and can be found at ``google.cloud.bigquery.magics.context``.
    """

    def __init__(self):
        self._credentials = None
        self._project = None
        self._connection = None
        self._default_query_job_config = bigquery.QueryJobConfig()
        self._bigquery_client_options = client_options.ClientOptions()
        self._bqstorage_client_options = client_options.ClientOptions()
        self._progress_bar_type = "tqdm_notebook"

    @property
    def credentials(self):
        """google.auth.credentials.Credentials: Credentials to use for queries
        performed through IPython magics.

        Note:
            These credentials do not need to be explicitly defined if you are
            using Application Default Credentials. If you are not using
            Application Default Credentials, manually construct a
            :class:`google.auth.credentials.Credentials` object and set it as
            the context credentials as demonstrated in the example below. See
            `auth docs`_ for more information on obtaining credentials.

        Example:
            Manually setting the context credentials:

            >>> from google.cloud.bigquery import magics
            >>> from google.oauth2 import service_account
            >>> credentials = (service_account
            ...     .Credentials.from_service_account_file(
            ...         '/path/to/key.json'))
            >>> magics.context.credentials = credentials


        .. _auth docs: http://google-auth.readthedocs.io
            /en/latest/user-guide.html#obtaining-credentials
        """
        if self._credentials is None:
            self._credentials, _ = google.auth.default()
        return self._credentials

    @credentials.setter
    def credentials(self, value):
        self._credentials = value

    @property
    def project(self):
        """str: Default project to use for queries performed through IPython
        magics.

        Note:
            The project does not need to be explicitly defined if you have an
            environment default project set. If you do not have a default
            project set in your environment, manually assign the project as
            demonstrated in the example below.

        Example:
            Manually setting the context project:

            >>> from google.cloud.bigquery import magics
            >>> magics.context.project = 'my-project'
        """
        if self._project is None:
            _, self._project = google.auth.default()
        return self._project

    @project.setter
    def project(self, value):
        self._project = value

    @property
    def bigquery_client_options(self):
        """google.api_core.client_options.ClientOptions: client options to be
        used through IPython magics.

        Note::
            The client options do not need to be explicitly defined if no
            special network connections are required. Normally you would be
            using the https://bigquery.googleapis.com/ end point.

        Example:
            Manually setting the endpoint:

            >>> from google.cloud.bigquery import magics
            >>> client_options = {}
            >>> client_options['api_endpoint'] = "https://some.special.url"
            >>> magics.context.bigquery_client_options = client_options
        """
        return self._bigquery_client_options

    @bigquery_client_options.setter
    def bigquery_client_options(self, value):
        self._bigquery_client_options = value

    @property
    def bqstorage_client_options(self):
        """google.api_core.client_options.ClientOptions: client options to be
        used through IPython magics for the storage client.

        Note::
            The client options do not need to be explicitly defined if no
            special network connections are required. Normally you would be
            using the https://bigquerystorage.googleapis.com/ end point.

        Example:
            Manually setting the endpoint:

            >>> from google.cloud.bigquery import magics
            >>> client_options = {}
            >>> client_options['api_endpoint'] = "https://some.special.url"
            >>> magics.context.bqstorage_client_options = client_options
        """
        return self._bqstorage_client_options

    @bqstorage_client_options.setter
    def bqstorage_client_options(self, value):
        self._bqstorage_client_options = value

    @property
    def default_query_job_config(self):
        """google.cloud.bigquery.job.QueryJobConfig: Default job
        configuration for queries.

        The context's :class:`~google.cloud.bigquery.job.QueryJobConfig` is
        used for queries. Some properties can be overridden with arguments to
        the magics.

        Example:
            Manually setting the default value for ``maximum_bytes_billed``
            to 100 MB:

            >>> from google.cloud.bigquery import magics
            >>> magics.context.default_query_job_config.maximum_bytes_billed = 100000000
        """
        return self._default_query_job_config

    @default_query_job_config.setter
    def default_query_job_config(self, value):
        self._default_query_job_config = value

    @property
    def progress_bar_type(self):
        """str: Default progress bar type to use to display progress bar while
        executing queries through IPython magics.

        Note::
            Install the ``tqdm`` package to use this feature.

        Example:
            Manually setting the progress_bar_type:

            >>> from google.cloud.bigquery import magics
            >>> magics.context.progress_bar_type = "tqdm_notebook"
        """
        return self._progress_bar_type

    @progress_bar_type.setter
    def progress_bar_type(self, value):
        self._progress_bar_type = value


# If bigquery_magics is available, we load that extension rather than this one.
# Ensure google.cloud.bigquery.magics.context setters are on the correct magics
# implementation in case the user has installed the package but hasn't updated
# their code.
if bigquery_magics is not None:
    context = bigquery_magics.context
else:
    context = Context()


def _handle_error(error, destination_var=None):
    """Process a query execution error.

    Args:
        error (Exception):
            An exception that occurred during the query execution.
        destination_var (Optional[str]):
            The name of the IPython session variable to store the query job.
    """
    if destination_var:
        query_job = getattr(error, "query_job", None)

        if query_job is not None:
            IPython.get_ipython().push({destination_var: query_job})
        else:
            # this is the case when previewing table rows by providing just
            # table ID to cell magic
            print(
                "Could not save output to variable '{}'.".format(destination_var),
                file=sys.stderr,
            )

    print("\nERROR:\n", str(error), file=sys.stderr)


def _run_query(client, query, job_config=None):
    """Runs a query while printing status updates

    Args:
        client (google.cloud.bigquery.client.Client):
            Client to bundle configuration needed for API requests.
        query (str):
            SQL query to be executed. Defaults to the standard SQL dialect.
            Use the ``job_config`` parameter to change dialects.
        job_config (Optional[google.cloud.bigquery.job.QueryJobConfig]):
            Extra configuration options for the job.

    Returns:
        google.cloud.bigquery.job.QueryJob: the query job created

    Example:
        >>> client = bigquery.Client()
        >>> _run_query(client, "SELECT 17")
        Executing query with job ID: bf633912-af2c-4780-b568-5d868058632b
        Query executing: 1.66s
        Query complete after 2.07s
        'bf633912-af2c-4780-b568-5d868058632b'
    """
    start_time = time.perf_counter()
    query_job = client.query(query, job_config=job_config)

    if job_config and job_config.dry_run:
        return query_job

    print(f"Executing query with job ID: {query_job.job_id}")

    while True:
        print(
            f"\rQuery executing: {time.perf_counter() - start_time:.2f}s".format(),
            end="",
        )
        try:
            query_job.result(timeout=0.5)
            break
        except futures.TimeoutError:
            continue
    print(f"\nJob ID {query_job.job_id} successfully executed")
    return query_job


def _create_dataset_if_necessary(client, dataset_id):
    """Create a dataset in the current project if it doesn't exist.

    Args:
        client (google.cloud.bigquery.client.Client):
            Client to bundle configuration needed for API requests.
        dataset_id (str):
            Dataset id.
    """
    dataset_reference = bigquery.dataset.DatasetReference(client.project, dataset_id)
    try:
        dataset = client.get_dataset(dataset_reference)
        return
    except NotFound:
        pass
    dataset = bigquery.Dataset(dataset_reference)
    dataset.location = client.location
    print(f"Creating dataset: {dataset_id}")
    dataset = client.create_dataset(dataset)


@magic_arguments.magic_arguments()
@magic_arguments.argument(
    "destination_var",
    nargs="?",
    help=("If provided, save the output to this variable instead of displaying it."),
)
@magic_arguments.argument(
    "--destination_table",
    type=str,
    default=None,
    help=(
        "If provided, save the output of the query to a new BigQuery table. "
        "Variable should be in a format <dataset_id>.<table_id>. "
        "If table does not exists, it will be created. "
        "If table already exists, its data will be overwritten."
    ),
)
@magic_arguments.argument(
    "--project",
    type=str,
    default=None,
    help=("Project to use for executing this query. Defaults to the context project."),
)
@magic_arguments.argument(
    "--max_results",
    default=None,
    help=(
        "Maximum number of rows in dataframe returned from executing the query."
        "Defaults to returning all rows."
    ),
)
@magic_arguments.argument(
    "--maximum_bytes_billed",
    default=None,
    help=(
        "maximum_bytes_billed to use for executing this query. Defaults to "
        "the context default_query_job_config.maximum_bytes_billed."
    ),
)
@magic_arguments.argument(
    "--dry_run",
    action="store_true",
    default=False,
    help=(
        "Sets query to be a dry run to estimate costs. "
        "Defaults to executing the query instead of dry run if this argument is not used."
    ),
)
@magic_arguments.argument(
    "--use_legacy_sql",
    action="store_true",
    default=False,
    help=(
        "Sets query to use Legacy SQL instead of Standard SQL. Defaults to "
        "Standard SQL if this argument is not used."
    ),
)
@magic_arguments.argument(
    "--bigquery_api_endpoint",
    type=str,
    default=None,
    help=(
        "The desired API endpoint, e.g., bigquery.googlepis.com. Defaults to this "
        "option's value in the context bigquery_client_options."
    ),
)
@magic_arguments.argument(
    "--bqstorage_api_endpoint",
    type=str,
    default=None,
    help=(
        "The desired API endpoint, e.g., bigquerystorage.googlepis.com. Defaults to "
        "this option's value in the context bqstorage_client_options."
    ),
)
@magic_arguments.argument(
    "--no_query_cache",
    action="store_true",
    default=False,
    help=("Do not use cached query results."),
)
@magic_arguments.argument(
    "--use_bqstorage_api",
    action="store_true",
    default=None,
    help=(
        "[Deprecated] The BigQuery Storage API is already used by default to "
        "download large query results, and this option has no effect. "
        "If you want to switch to the classic REST API instead, use the "
        "--use_rest_api option."
    ),
)
@magic_arguments.argument(
    "--use_rest_api",
    action="store_true",
    default=False,
    help=(
        "Use the classic REST API instead of the BigQuery Storage API to "
        "download query results."
    ),
)
@magic_arguments.argument(
    "--verbose",
    action="store_true",
    default=False,
    help=(
        "If set, print verbose output, including the query job ID and the "
        "amount of time for the query to finish. By default, this "
        "information will be displayed as the query runs, but will be "
        "cleared after the query is finished."
    ),
)
@magic_arguments.argument(
    "--params",
    nargs="+",
    default=None,
    help=(
        "Parameters to format the query string. If present, the --params "
        "flag should be followed by a string representation of a dictionary "
        "in the format {'param_name': 'param_value'} (ex. {\"num\": 17}), "
        "or a reference to a dictionary in the same format. The dictionary "
        "reference can be made by including a '$' before the variable "
        "name (ex. $my_dict_var)."
    ),
)
@magic_arguments.argument(
    "--progress_bar_type",
    type=str,
    default=None,
    help=(
        "Sets progress bar type to display a progress bar while executing the query."
        "Defaults to use tqdm_notebook. Install the ``tqdm`` package to use this feature."
    ),
)
@magic_arguments.argument(
    "--location",
    type=str,
    default=None,
    help=(
        "Set the location to execute query."
        "Defaults to location set in query setting in console."
    ),
)
def _cell_magic(line, query):
    """Underlying function for bigquery cell magic

    Note:
        This function contains the underlying logic for the 'bigquery' cell
        magic. This function is not meant to be called directly.

    Args:
        line (str): "%%bigquery" followed by arguments as required
        query (str): SQL query to run

    Returns:
        pandas.DataFrame: the query results.
    """
    # The built-in parser does not recognize Python structures such as dicts, thus
    # we extract the "--params" option and inteprpret it separately.
    try:
        params_option_value, rest_of_args = _split_args_line(line)
    except lap.exceptions.QueryParamsParseError as exc:
        rebranded_error = SyntaxError(
            "--params is not a correctly formatted JSON string or a JSON "
            "serializable dictionary"
        )
        raise rebranded_error from exc
    except lap.exceptions.DuplicateQueryParamsError as exc:
        rebranded_error = ValueError("Duplicate --params option.")
        raise rebranded_error from exc
    except lap.exceptions.ParseError as exc:
        rebranded_error = ValueError(
            "Unrecognized input, are option values correct? "
            "Error details: {}".format(exc.args[0])
        )
        raise rebranded_error from exc

    args = magic_arguments.parse_argstring(_cell_magic, rest_of_args)

    if args.use_bqstorage_api is not None:
        warnings.warn(
            "Deprecated option --use_bqstorage_api, the BigQuery "
            "Storage API is already used by default.",
            category=DeprecationWarning,
        )
    use_bqstorage_api = not args.use_rest_api
    location = args.location

    params = []
    if params_option_value:
        # A non-existing params variable is not expanded and ends up in the input
        # in its raw form, e.g. "$query_params".
        if params_option_value.startswith("$"):
            msg = 'Parameter expansion failed, undefined variable "{}".'.format(
                params_option_value[1:]
            )
            raise NameError(msg)

        params = _helpers.to_query_parameters(ast.literal_eval(params_option_value), {})

    project = args.project or context.project

    bigquery_client_options = copy.deepcopy(context.bigquery_client_options)
    if args.bigquery_api_endpoint:
        if isinstance(bigquery_client_options, dict):
            bigquery_client_options["api_endpoint"] = args.bigquery_api_endpoint
        else:
            bigquery_client_options.api_endpoint = args.bigquery_api_endpoint

    client = bigquery.Client(
        project=project,
        credentials=context.credentials,
        default_query_job_config=context.default_query_job_config,
        client_info=client_info.ClientInfo(user_agent=IPYTHON_USER_AGENT),
        client_options=bigquery_client_options,
        location=location,
    )
    if context._connection:
        client._connection = context._connection

    bqstorage_client_options = copy.deepcopy(context.bqstorage_client_options)
    if args.bqstorage_api_endpoint:
        if isinstance(bqstorage_client_options, dict):
            bqstorage_client_options["api_endpoint"] = args.bqstorage_api_endpoint
        else:
            bqstorage_client_options.api_endpoint = args.bqstorage_api_endpoint

    bqstorage_client = _make_bqstorage_client(
        client,
        use_bqstorage_api,
        bqstorage_client_options,
    )

    close_transports = functools.partial(_close_transports, client, bqstorage_client)

    try:
        if args.max_results:
            max_results = int(args.max_results)
        else:
            max_results = None

        query = query.strip()

        if not query:
            error = ValueError("Query is missing.")
            _handle_error(error, args.destination_var)
            return

        # Check if query is given as a reference to a variable.
        if query.startswith("$"):
            query_var_name = query[1:]

            if not query_var_name:
                missing_msg = 'Missing query variable name, empty "$" is not allowed.'
                raise NameError(missing_msg)

            if query_var_name.isidentifier():
                ip = IPython.get_ipython()
                query = ip.user_ns.get(query_var_name, ip)  # ip serves as a sentinel

                if query is ip:
                    raise NameError(
                        f"Unknown query, variable {query_var_name} does not exist."
                    )
                else:
                    if not isinstance(query, (str, bytes)):
                        raise TypeError(
                            f"Query variable {query_var_name} must be a string "
                            "or a bytes-like value."
                        )

        # Any query that does not contain whitespace (aside from leading and trailing whitespace)
        # is assumed to be a table id
        if not re.search(r"\s", query):
            try:
                rows = client.list_rows(query, max_results=max_results)
            except Exception as ex:
                _handle_error(ex, args.destination_var)
                return

            result = rows.to_dataframe(
                bqstorage_client=bqstorage_client,
                create_bqstorage_client=False,
            )
            if args.destination_var:
                IPython.get_ipython().push({args.destination_var: result})
                return
            else:
                return result

        job_config = bigquery.job.QueryJobConfig()
        job_config.query_parameters = params
        job_config.use_legacy_sql = args.use_legacy_sql
        job_config.dry_run = args.dry_run

        # Don't override context job config unless --no_query_cache is explicitly set.
        if args.no_query_cache:
            job_config.use_query_cache = False

        if args.destination_table:
            split = args.destination_table.split(".")
            if len(split) != 2:
                raise ValueError(
                    "--destination_table should be in a <dataset_id>.<table_id> format."
                )
            dataset_id, table_id = split
            job_config.allow_large_results = True
            dataset_ref = bigquery.dataset.DatasetReference(client.project, dataset_id)
            destination_table_ref = dataset_ref.table(table_id)
            job_config.destination = destination_table_ref
            job_config.create_disposition = "CREATE_IF_NEEDED"
            job_config.write_disposition = "WRITE_TRUNCATE"
            _create_dataset_if_necessary(client, dataset_id)

        if args.maximum_bytes_billed == "None":
            job_config.maximum_bytes_billed = 0
        elif args.maximum_bytes_billed is not None:
            value = int(args.maximum_bytes_billed)
            job_config.maximum_bytes_billed = value

        try:
            query_job = _run_query(client, query, job_config=job_config)
        except Exception as ex:
            _handle_error(ex, args.destination_var)
            return

        if not args.verbose:
            display.clear_output()

        if args.dry_run and args.destination_var:
            IPython.get_ipython().push({args.destination_var: query_job})
            return
        elif args.dry_run:
            print(
                "Query validated. This query will process {} bytes.".format(
                    query_job.total_bytes_processed
                )
            )
            return query_job

        progress_bar = context.progress_bar_type or args.progress_bar_type

        if max_results:
            result = query_job.result(max_results=max_results).to_dataframe(
                bqstorage_client=None,
                create_bqstorage_client=False,
                progress_bar_type=progress_bar,
            )
        else:
            result = query_job.to_dataframe(
                bqstorage_client=bqstorage_client,
                create_bqstorage_client=False,
                progress_bar_type=progress_bar,
            )

        if args.destination_var:
            IPython.get_ipython().push({args.destination_var: result})
        else:
            return result
    finally:
        close_transports()


def _split_args_line(line):
    """Split out the --params option value from the input line arguments.

    Args:
        line (str): The line arguments passed to the cell magic.

    Returns:
        Tuple[str, str]
    """
    lexer = lap.Lexer(line)
    scanner = lap.Parser(lexer)
    tree = scanner.input_line()

    extractor = lap.QueryParamsExtractor()
    params_option_value, rest_of_args = extractor.visit(tree)

    return params_option_value, rest_of_args


def _make_bqstorage_client(client, use_bqstorage_api, client_options):
    """Creates a BigQuery Storage client.

    Args:
        client (:class:`~google.cloud.bigquery.client.Client`): BigQuery client.
        use_bqstorage_api (bool): whether BigQuery Storage API is used or not.
        client_options (:class:`google.api_core.client_options.ClientOptions`):
            Custom options used with a new BigQuery Storage client instance
            if one is created.

    Raises:
        ImportError: if google-cloud-bigquery-storage is not installed, or
            grpcio package is not installed.


    Returns:
        None: if ``use_bqstorage_api == False``, or google-cloud-bigquery-storage
            is outdated.
        BigQuery Storage Client:
    """
    if not use_bqstorage_api:
        return None

    try:
        _versions_helpers.BQ_STORAGE_VERSIONS.try_import(raise_if_error=True)
    except exceptions.BigQueryStorageNotFoundError as err:
        customized_error = ImportError(
            "The default BigQuery Storage API client cannot be used, install "
            "the missing google-cloud-bigquery-storage and pyarrow packages "
            "to use it. Alternatively, use the classic REST API by specifying "
            "the --use_rest_api magic option."
        )
        raise customized_error from err
    except exceptions.LegacyBigQueryStorageError:
        pass

    try:
        from google.api_core.gapic_v1 import client_info as gapic_client_info
    except ImportError as err:
        customized_error = ImportError(
            "Install the grpcio package to use the BigQuery Storage API."
        )
        raise customized_error from err

    return client._ensure_bqstorage_client(
        client_options=client_options,
        client_info=gapic_client_info.ClientInfo(user_agent=IPYTHON_USER_AGENT),
    )


def _close_transports(client, bqstorage_client):
    """Close the given clients' underlying transport channels.

    Closing the transport is needed to release system resources, namely open
    sockets.

    Args:
        client (:class:`~google.cloud.bigquery.client.Client`):
        bqstorage_client
            (Optional[:class:`~google.cloud.bigquery_storage.BigQueryReadClient`]):
            A client for the BigQuery Storage API.

    """
    client.close()
    if bqstorage_client is not None:
        bqstorage_client._transport.grpc_channel.close()
