from abc import ABC, abstractmethod
import io
import os
import gzip
import socket
import ssl
import time
import warnings
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from urllib.request import getproxies

try:
    import brotli  # type: ignore
except ImportError:
    brotli = None

import urllib3
import certifi

import sentry_sdk
from sentry_sdk.consts import EndpointType
from sentry_sdk.utils import Dsn, logger, capture_internal_exceptions
from sentry_sdk.worker import BackgroundWorker
from sentry_sdk.envelope import Envelope, Item, PayloadRef

from typing import TYPE_CHECKING, cast, List, Dict

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import DefaultDict
    from typing import Iterable
    from typing import Mapping
    from typing import Optional
    from typing import Self
    from typing import Tuple
    from typing import Type
    from typing import Union

    from urllib3.poolmanager import PoolManager
    from urllib3.poolmanager import ProxyManager

    from sentry_sdk._types import Event, EventDataCategory

KEEP_ALIVE_SOCKET_OPTIONS = []
for option in [
    (socket.SOL_SOCKET, lambda: getattr(socket, "SO_KEEPALIVE"), 1),  # noqa: B009
    (socket.SOL_TCP, lambda: getattr(socket, "TCP_KEEPIDLE"), 45),  # noqa: B009
    (socket.SOL_TCP, lambda: getattr(socket, "TCP_KEEPINTVL"), 10),  # noqa: B009
    (socket.SOL_TCP, lambda: getattr(socket, "TCP_KEEPCNT"), 6),  # noqa: B009
]:
    try:
        KEEP_ALIVE_SOCKET_OPTIONS.append((option[0], option[1](), option[2]))
    except AttributeError:
        # a specific option might not be available on specific systems,
        # e.g. TCP_KEEPIDLE doesn't exist on macOS
        pass


class Transport(ABC):
    """Baseclass for all transports.

    A transport is used to send an event to sentry.
    """

    parsed_dsn = None  # type: Optional[Dsn]

    def __init__(self, options=None):
        # type: (Self, Optional[Dict[str, Any]]) -> None
        self.options = options
        if options and options["dsn"] is not None and options["dsn"]:
            self.parsed_dsn = Dsn(options["dsn"])
        else:
            self.parsed_dsn = None

    def capture_event(self, event):
        # type: (Self, Event) -> None
        """
        DEPRECATED: Please use capture_envelope instead.

        This gets invoked with the event dictionary when an event should
        be sent to sentry.
        """

        warnings.warn(
            "capture_event is deprecated, please use capture_envelope instead!",
            DeprecationWarning,
            stacklevel=2,
        )

        envelope = Envelope()
        envelope.add_event(event)
        self.capture_envelope(envelope)

    @abstractmethod
    def capture_envelope(self, envelope):
        # type: (Self, Envelope) -> None
        """
        Send an envelope to Sentry.

        Envelopes are a data container format that can hold any type of data
        submitted to Sentry. We use it to send all event data (including errors,
        transactions, crons check-ins, etc.) to Sentry.
        """
        pass

    def flush(
        self,
        timeout,
        callback=None,
    ):
        # type: (Self, float, Optional[Any]) -> None
        """
        Wait `timeout` seconds for the current events to be sent out.

        The default implementation is a no-op, since this method may only be relevant to some transports.
        Subclasses should override this method if necessary.
        """
        return None

    def kill(self):
        # type: (Self) -> None
        """
        Forcefully kills the transport.

        The default implementation is a no-op, since this method may only be relevant to some transports.
        Subclasses should override this method if necessary.
        """
        return None

    def record_lost_event(
        self,
        reason,  # type: str
        data_category=None,  # type: Optional[EventDataCategory]
        item=None,  # type: Optional[Item]
        *,
        quantity=1,  # type: int
    ):
        # type: (...) -> None
        """This increments a counter for event loss by reason and
        data category by the given positive-int quantity (default 1).

        If an item is provided, the data category and quantity are
        extracted from the item, and the values passed for
        data_category and quantity are ignored.

        When recording a lost transaction via data_category="transaction",
        the calling code should also record the lost spans via this method.
        When recording lost spans, `quantity` should be set to the number
        of contained spans, plus one for the transaction itself. When
        passing an Item containing a transaction via the `item` parameter,
        this method automatically records the lost spans.
        """
        return None

    def is_healthy(self):
        # type: (Self) -> bool
        return True

    def __del__(self):
        # type: (Self) -> None
        try:
            self.kill()
        except Exception:
            pass


def _parse_rate_limits(header, now=None):
    # type: (str, Optional[datetime]) -> Iterable[Tuple[Optional[EventDataCategory], datetime]]
    if now is None:
        now = datetime.now(timezone.utc)

    for limit in header.split(","):
        try:
            parameters = limit.strip().split(":")
            retry_after_val, categories = parameters[:2]

            retry_after = now + timedelta(seconds=int(retry_after_val))
            for category in categories and categories.split(";") or (None,):
                if category == "metric_bucket":
                    try:
                        namespaces = parameters[4].split(";")
                    except IndexError:
                        namespaces = []

                    if not namespaces or "custom" in namespaces:
                        yield category, retry_after  # type: ignore

                else:
                    yield category, retry_after  # type: ignore
        except (LookupError, ValueError):
            continue


class BaseHttpTransport(Transport):
    """The base HTTP transport."""

    def __init__(self, options):
        # type: (Self, Dict[str, Any]) -> None
        from sentry_sdk.consts import VERSION

        Transport.__init__(self, options)
        assert self.parsed_dsn is not None
        self.options = options  # type: Dict[str, Any]
        self._worker = BackgroundWorker(queue_size=options["transport_queue_size"])
        self._auth = self.parsed_dsn.to_auth("sentry.python/%s" % VERSION)
        self._disabled_until = {}  # type: Dict[Optional[EventDataCategory], datetime]
        # We only use this Retry() class for the `get_retry_after` method it exposes
        self._retry = urllib3.util.Retry()
        self._discarded_events = defaultdict(
            int
        )  # type: DefaultDict[Tuple[EventDataCategory, str], int]
        self._last_client_report_sent = time.time()

        self._pool = self._make_pool()

        # Backwards compatibility for deprecated `self.hub_class` attribute
        self._hub_cls = sentry_sdk.Hub

        experiments = options.get("_experiments", {})
        compression_level = experiments.get(
            "transport_compression_level",
            experiments.get("transport_zlib_compression_level"),
        )
        compression_algo = experiments.get(
            "transport_compression_algo",
            (
                "gzip"
                # if only compression level is set, assume gzip for backwards compatibility
                # if we don't have brotli available, fallback to gzip
                if compression_level is not None or brotli is None
                else "br"
            ),
        )

        if compression_algo == "br" and brotli is None:
            logger.warning(
                "You asked for brotli compression without the Brotli module, falling back to gzip -9"
            )
            compression_algo = "gzip"
            compression_level = None

        if compression_algo not in ("br", "gzip"):
            logger.warning(
                "Unknown compression algo %s, disabling compression", compression_algo
            )
            self._compression_level = 0
            self._compression_algo = None
        else:
            self._compression_algo = compression_algo

        if compression_level is not None:
            self._compression_level = compression_level
        elif self._compression_algo == "gzip":
            self._compression_level = 9
        elif self._compression_algo == "br":
            self._compression_level = 4

    def record_lost_event(
        self,
        reason,  # type: str
        data_category=None,  # type: Optional[EventDataCategory]
        item=None,  # type: Optional[Item]
        *,
        quantity=1,  # type: int
    ):
        # type: (...) -> None
        if not self.options["send_client_reports"]:
            return

        if item is not None:
            data_category = item.data_category
            quantity = 1  # If an item is provided, we always count it as 1 (except for attachments, handled below).

            if data_category == "transaction":
                # Also record the lost spans
                event = item.get_transaction_event() or {}

                # +1 for the transaction itself
                span_count = (
                    len(cast(List[Dict[str, object]], event.get("spans") or [])) + 1
                )
                self.record_lost_event(reason, "span", quantity=span_count)

            elif data_category == "attachment":
                # quantity of 0 is actually 1 as we do not want to count
                # empty attachments as actually empty.
                quantity = len(item.get_bytes()) or 1

        elif data_category is None:
            raise TypeError("data category not provided")

        self._discarded_events[data_category, reason] += quantity

    def _get_header_value(self, response, header):
        # type: (Self, Any, str) -> Optional[str]
        return response.headers.get(header)

    def _update_rate_limits(self, response):
        # type: (Self, Union[urllib3.BaseHTTPResponse, httpcore.Response]) -> None

        # new sentries with more rate limit insights.  We honor this header
        # no matter of the status code to update our internal rate limits.
        header = self._get_header_value(response, "x-sentry-rate-limits")
        if header:
            logger.warning("Rate-limited via x-sentry-rate-limits")
            self._disabled_until.update(_parse_rate_limits(header))

        # old sentries only communicate global rate limit hits via the
        # retry-after header on 429.  This header can also be emitted on new
        # sentries if a proxy in front wants to globally slow things down.
        elif response.status == 429:
            logger.warning("Rate-limited via 429")
            retry_after_value = self._get_header_value(response, "Retry-After")
            retry_after = (
                self._retry.parse_retry_after(retry_after_value)
                if retry_after_value is not None
                else None
            ) or 60
            self._disabled_until[None] = datetime.now(timezone.utc) + timedelta(
                seconds=retry_after
            )

    def _send_request(
        self,
        body,
        headers,
        endpoint_type=EndpointType.ENVELOPE,
        envelope=None,
    ):
        # type: (Self, bytes, Dict[str, str], EndpointType, Optional[Envelope]) -> None

        def record_loss(reason):
            # type: (str) -> None
            if envelope is None:
                self.record_lost_event(reason, data_category="error")
            else:
                for item in envelope.items:
                    self.record_lost_event(reason, item=item)

        headers.update(
            {
                "User-Agent": str(self._auth.client),
                "X-Sentry-Auth": str(self._auth.to_header()),
            }
        )
        try:
            response = self._request(
                "POST",
                endpoint_type,
                body,
                headers,
            )
        except Exception:
            self.on_dropped_event("network")
            record_loss("network_error")
            raise

        try:
            self._update_rate_limits(response)

            if response.status == 429:
                # if we hit a 429.  Something was rate limited but we already
                # acted on this in `self._update_rate_limits`.  Note that we
                # do not want to record event loss here as we will have recorded
                # an outcome in relay already.
                self.on_dropped_event("status_429")
                pass

            elif response.status >= 300 or response.status < 200:
                logger.error(
                    "Unexpected status code: %s (body: %s)",
                    response.status,
                    getattr(response, "data", getattr(response, "content", None)),
                )
                self.on_dropped_event("status_{}".format(response.status))
                record_loss("network_error")
        finally:
            response.close()

    def on_dropped_event(self, _reason):
        # type: (Self, str) -> None
        return None

    def _fetch_pending_client_report(self, force=False, interval=60):
        # type: (Self, bool, int) -> Optional[Item]
        if not self.options["send_client_reports"]:
            return None

        if not (force or self._last_client_report_sent < time.time() - interval):
            return None

        discarded_events = self._discarded_events
        self._discarded_events = defaultdict(int)
        self._last_client_report_sent = time.time()

        if not discarded_events:
            return None

        return Item(
            PayloadRef(
                json={
                    "timestamp": time.time(),
                    "discarded_events": [
                        {"reason": reason, "category": category, "quantity": quantity}
                        for (
                            (category, reason),
                            quantity,
                        ) in discarded_events.items()
                    ],
                }
            ),
            type="client_report",
        )

    def _flush_client_reports(self, force=False):
        # type: (Self, bool) -> None
        client_report = self._fetch_pending_client_report(force=force, interval=60)
        if client_report is not None:
            self.capture_envelope(Envelope(items=[client_report]))

    def _check_disabled(self, category):
        # type: (str) -> bool
        def _disabled(bucket):
            # type: (Any) -> bool

            # The envelope item type used for metrics is statsd
            # whereas the rate limit category is metric_bucket
            if bucket == "statsd":
                bucket = "metric_bucket"

            ts = self._disabled_until.get(bucket)
            return ts is not None and ts > datetime.now(timezone.utc)

        return _disabled(category) or _disabled(None)

    def _is_rate_limited(self):
        # type: (Self) -> bool
        return any(
            ts > datetime.now(timezone.utc) for ts in self._disabled_until.values()
        )

    def _is_worker_full(self):
        # type: (Self) -> bool
        return self._worker.full()

    def is_healthy(self):
        # type: (Self) -> bool
        return not (self._is_worker_full() or self._is_rate_limited())

    def _send_envelope(self, envelope):
        # type: (Self, Envelope) -> None

        # remove all items from the envelope which are over quota
        new_items = []
        for item in envelope.items:
            if self._check_disabled(item.data_category):
                if item.data_category in ("transaction", "error", "default", "statsd"):
                    self.on_dropped_event("self_rate_limits")
                self.record_lost_event("ratelimit_backoff", item=item)
            else:
                new_items.append(item)

        # Since we're modifying the envelope here make a copy so that others
        # that hold references do not see their envelope modified.
        envelope = Envelope(headers=envelope.headers, items=new_items)

        if not envelope.items:
            return None

        # since we're already in the business of sending out an envelope here
        # check if we have one pending for the stats session envelopes so we
        # can attach it to this enveloped scheduled for sending.  This will
        # currently typically attach the client report to the most recent
        # session update.
        client_report_item = self._fetch_pending_client_report(interval=30)
        if client_report_item is not None:
            envelope.items.append(client_report_item)

        content_encoding, body = self._serialize_envelope(envelope)

        assert self.parsed_dsn is not None
        logger.debug(
            "Sending envelope [%s] project:%s host:%s",
            envelope.description,
            self.parsed_dsn.project_id,
            self.parsed_dsn.host,
        )

        headers = {
            "Content-Type": "application/x-sentry-envelope",
        }
        if content_encoding:
            headers["Content-Encoding"] = content_encoding

        self._send_request(
            body.getvalue(),
            headers=headers,
            endpoint_type=EndpointType.ENVELOPE,
            envelope=envelope,
        )
        return None

    def _serialize_envelope(self, envelope):
        # type: (Self, Envelope) -> tuple[Optional[str], io.BytesIO]
        content_encoding = None
        body = io.BytesIO()
        if self._compression_level == 0 or self._compression_algo is None:
            envelope.serialize_into(body)
        else:
            content_encoding = self._compression_algo
            if self._compression_algo == "br" and brotli is not None:
                body.write(
                    brotli.compress(
                        envelope.serialize(), quality=self._compression_level
                    )
                )
            else:  # assume gzip as we sanitize the algo value in init
                with gzip.GzipFile(
                    fileobj=body, mode="w", compresslevel=self._compression_level
                ) as f:
                    envelope.serialize_into(f)

        return content_encoding, body

    def _get_pool_options(self):
        # type: (Self) -> Dict[str, Any]
        raise NotImplementedError()

    def _in_no_proxy(self, parsed_dsn):
        # type: (Self, Dsn) -> bool
        no_proxy = getproxies().get("no")
        if not no_proxy:
            return False
        for host in no_proxy.split(","):
            host = host.strip()
            if parsed_dsn.host.endswith(host) or parsed_dsn.netloc.endswith(host):
                return True
        return False

    def _make_pool(self):
        # type: (Self) -> Union[PoolManager, ProxyManager, httpcore.SOCKSProxy, httpcore.HTTPProxy, httpcore.ConnectionPool]
        raise NotImplementedError()

    def _request(
        self,
        method,
        endpoint_type,
        body,
        headers,
    ):
        # type: (Self, str, EndpointType, Any, Mapping[str, str]) -> Union[urllib3.BaseHTTPResponse, httpcore.Response]
        raise NotImplementedError()

    def capture_envelope(
        self, envelope  # type: Envelope
    ):
        # type: (...) -> None
        def send_envelope_wrapper():
            # type: () -> None
            with capture_internal_exceptions():
                self._send_envelope(envelope)
                self._flush_client_reports()

        if not self._worker.submit(send_envelope_wrapper):
            self.on_dropped_event("full_queue")
            for item in envelope.items:
                self.record_lost_event("queue_overflow", item=item)

    def flush(
        self,
        timeout,
        callback=None,
    ):
        # type: (Self, float, Optional[Callable[[int, float], None]]) -> None
        logger.debug("Flushing HTTP transport")

        if timeout > 0:
            self._worker.submit(lambda: self._flush_client_reports(force=True))
            self._worker.flush(timeout, callback)

    def kill(self):
        # type: (Self) -> None
        logger.debug("Killing HTTP transport")
        self._worker.kill()

    @staticmethod
    def _warn_hub_cls():
        # type: () -> None
        """Convenience method to warn users about the deprecation of the `hub_cls` attribute."""
        warnings.warn(
            "The `hub_cls` attribute is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )

    @property
    def hub_cls(self):
        # type: (Self) -> type[sentry_sdk.Hub]
        """DEPRECATED: This attribute is deprecated and will be removed in a future release."""
        HttpTransport._warn_hub_cls()
        return self._hub_cls

    @hub_cls.setter
    def hub_cls(self, value):
        # type: (Self, type[sentry_sdk.Hub]) -> None
        """DEPRECATED: This attribute is deprecated and will be removed in a future release."""
        HttpTransport._warn_hub_cls()
        self._hub_cls = value


class HttpTransport(BaseHttpTransport):
    if TYPE_CHECKING:
        _pool: Union[PoolManager, ProxyManager]

    def _get_pool_options(self):
        # type: (Self) -> Dict[str, Any]

        num_pools = self.options.get("_experiments", {}).get("transport_num_pools")
        options = {
            "num_pools": 2 if num_pools is None else int(num_pools),
            "cert_reqs": "CERT_REQUIRED",
        }

        socket_options = None  # type: Optional[List[Tuple[int, int, int | bytes]]]

        if self.options["socket_options"] is not None:
            socket_options = self.options["socket_options"]

        if self.options["keep_alive"]:
            if socket_options is None:
                socket_options = []

            used_options = {(o[0], o[1]) for o in socket_options}
            for default_option in KEEP_ALIVE_SOCKET_OPTIONS:
                if (default_option[0], default_option[1]) not in used_options:
                    socket_options.append(default_option)

        if socket_options is not None:
            options["socket_options"] = socket_options

        options["ca_certs"] = (
            self.options["ca_certs"]  # User-provided bundle from the SDK init
            or os.environ.get("SSL_CERT_FILE")
            or os.environ.get("REQUESTS_CA_BUNDLE")
            or certifi.where()
        )

        options["cert_file"] = self.options["cert_file"] or os.environ.get(
            "CLIENT_CERT_FILE"
        )
        options["key_file"] = self.options["key_file"] or os.environ.get(
            "CLIENT_KEY_FILE"
        )

        return options

    def _make_pool(self):
        # type: (Self) -> Union[PoolManager, ProxyManager]
        if self.parsed_dsn is None:
            raise ValueError("Cannot create HTTP-based transport without valid DSN")

        proxy = None
        no_proxy = self._in_no_proxy(self.parsed_dsn)

        # try HTTPS first
        https_proxy = self.options["https_proxy"]
        if self.parsed_dsn.scheme == "https" and (https_proxy != ""):
            proxy = https_proxy or (not no_proxy and getproxies().get("https"))

        # maybe fallback to HTTP proxy
        http_proxy = self.options["http_proxy"]
        if not proxy and (http_proxy != ""):
            proxy = http_proxy or (not no_proxy and getproxies().get("http"))

        opts = self._get_pool_options()

        if proxy:
            proxy_headers = self.options["proxy_headers"]
            if proxy_headers:
                opts["proxy_headers"] = proxy_headers

            if proxy.startswith("socks"):
                use_socks_proxy = True
                try:
                    # Check if PySocks dependency is available
                    from urllib3.contrib.socks import SOCKSProxyManager
                except ImportError:
                    use_socks_proxy = False
                    logger.warning(
                        "You have configured a SOCKS proxy (%s) but support for SOCKS proxies is not installed. Disabling proxy support. Please add `PySocks` (or `urllib3` with the `[socks]` extra) to your dependencies.",
                        proxy,
                    )

                if use_socks_proxy:
                    return SOCKSProxyManager(proxy, **opts)
                else:
                    return urllib3.PoolManager(**opts)
            else:
                return urllib3.ProxyManager(proxy, **opts)
        else:
            return urllib3.PoolManager(**opts)

    def _request(
        self,
        method,
        endpoint_type,
        body,
        headers,
    ):
        # type: (Self, str, EndpointType, Any, Mapping[str, str]) -> urllib3.BaseHTTPResponse
        return self._pool.request(
            method,
            self._auth.get_api_url(endpoint_type),
            body=body,
            headers=headers,
        )


try:
    import httpcore
    import h2  # noqa: F401
except ImportError:
    # Sorry, no Http2Transport for you
    class Http2Transport(HttpTransport):
        def __init__(self, options):
            # type: (Self, Dict[str, Any]) -> None
            super().__init__(options)
            logger.warning(
                "You tried to use HTTP2Transport but don't have httpcore[http2] installed. Falling back to HTTPTransport."
            )

else:

    class Http2Transport(BaseHttpTransport):  # type: ignore
        """The HTTP2 transport based on httpcore."""

        if TYPE_CHECKING:
            _pool: Union[
                httpcore.SOCKSProxy, httpcore.HTTPProxy, httpcore.ConnectionPool
            ]

        def _get_header_value(self, response, header):
            # type: (Self, httpcore.Response, str) -> Optional[str]
            return next(
                (
                    val.decode("ascii")
                    for key, val in response.headers
                    if key.decode("ascii").lower() == header
                ),
                None,
            )

        def _request(
            self,
            method,
            endpoint_type,
            body,
            headers,
        ):
            # type: (Self, str, EndpointType, Any, Mapping[str, str]) -> httpcore.Response
            response = self._pool.request(
                method,
                self._auth.get_api_url(endpoint_type),
                content=body,
                headers=headers,  # type: ignore
            )
            return response

        def _get_pool_options(self):
            # type: (Self) -> Dict[str, Any]
            options = {
                "http2": self.parsed_dsn is not None
                and self.parsed_dsn.scheme == "https",
                "retries": 3,
            }  # type: Dict[str, Any]

            socket_options = (
                self.options["socket_options"]
                if self.options["socket_options"] is not None
                else []
            )

            used_options = {(o[0], o[1]) for o in socket_options}
            for default_option in KEEP_ALIVE_SOCKET_OPTIONS:
                if (default_option[0], default_option[1]) not in used_options:
                    socket_options.append(default_option)

            options["socket_options"] = socket_options

            ssl_context = ssl.create_default_context()
            ssl_context.load_verify_locations(
                self.options["ca_certs"]  # User-provided bundle from the SDK init
                or os.environ.get("SSL_CERT_FILE")
                or os.environ.get("REQUESTS_CA_BUNDLE")
                or certifi.where()
            )
            cert_file = self.options["cert_file"] or os.environ.get("CLIENT_CERT_FILE")
            key_file = self.options["key_file"] or os.environ.get("CLIENT_KEY_FILE")
            if cert_file is not None:
                ssl_context.load_cert_chain(cert_file, key_file)

            options["ssl_context"] = ssl_context

            return options

        def _make_pool(self):
            # type: (Self) -> Union[httpcore.SOCKSProxy, httpcore.HTTPProxy, httpcore.ConnectionPool]
            if self.parsed_dsn is None:
                raise ValueError("Cannot create HTTP-based transport without valid DSN")
            proxy = None
            no_proxy = self._in_no_proxy(self.parsed_dsn)

            # try HTTPS first
            https_proxy = self.options["https_proxy"]
            if self.parsed_dsn.scheme == "https" and (https_proxy != ""):
                proxy = https_proxy or (not no_proxy and getproxies().get("https"))

            # maybe fallback to HTTP proxy
            http_proxy = self.options["http_proxy"]
            if not proxy and (http_proxy != ""):
                proxy = http_proxy or (not no_proxy and getproxies().get("http"))

            opts = self._get_pool_options()

            if proxy:
                proxy_headers = self.options["proxy_headers"]
                if proxy_headers:
                    opts["proxy_headers"] = proxy_headers

                if proxy.startswith("socks"):
                    try:
                        if "socket_options" in opts:
                            socket_options = opts.pop("socket_options")
                            if socket_options:
                                logger.warning(
                                    "You have defined socket_options but using a SOCKS proxy which doesn't support these. We'll ignore socket_options."
                                )
                        return httpcore.SOCKSProxy(proxy_url=proxy, **opts)
                    except RuntimeError:
                        logger.warning(
                            "You have configured a SOCKS proxy (%s) but support for SOCKS proxies is not installed. Disabling proxy support.",
                            proxy,
                        )
                else:
                    return httpcore.HTTPProxy(proxy_url=proxy, **opts)

            return httpcore.ConnectionPool(**opts)


class _FunctionTransport(Transport):
    """
    DEPRECATED: Users wishing to provide a custom transport should subclass
    the Transport class, rather than providing a function.
    """

    def __init__(
        self, func  # type: Callable[[Event], None]
    ):
        # type: (...) -> None
        Transport.__init__(self)
        self._func = func

    def capture_event(
        self, event  # type: Event
    ):
        # type: (...) -> None
        self._func(event)
        return None

    def capture_envelope(self, envelope: Envelope) -> None:
        # Since function transports expect to be called with an event, we need
        # to iterate over the envelope and call the function for each event, via
        # the deprecated capture_event method.
        event = envelope.get_event()
        if event is not None:
            self.capture_event(event)


def make_transport(options):
    # type: (Dict[str, Any]) -> Optional[Transport]
    ref_transport = options["transport"]

    use_http2_transport = options.get("_experiments", {}).get("transport_http2", False)

    # By default, we use the http transport class
    transport_cls = (
        Http2Transport if use_http2_transport else HttpTransport
    )  # type: Type[Transport]

    if isinstance(ref_transport, Transport):
        return ref_transport
    elif isinstance(ref_transport, type) and issubclass(ref_transport, Transport):
        transport_cls = ref_transport
    elif callable(ref_transport):
        warnings.warn(
            "Function transports are deprecated and will be removed in a future release."
            "Please provide a Transport instance or subclass, instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _FunctionTransport(ref_transport)

    # if a transport class is given only instantiate it if the dsn is not
    # empty or None
    if options["dsn"]:
        return transport_cls(options)

    return None
