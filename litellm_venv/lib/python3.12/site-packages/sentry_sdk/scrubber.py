from sentry_sdk.utils import (
    capture_internal_exceptions,
    AnnotatedValue,
    iter_event_frames,
)

from typing import TYPE_CHECKING, cast, List, Dict

if TYPE_CHECKING:
    from sentry_sdk._types import Event
    from typing import Optional


DEFAULT_DENYLIST = [
    # stolen from relay
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "auth",
    "credentials",
    "mysql_pwd",
    "privatekey",
    "private_key",
    "token",
    "session",
    # django
    "csrftoken",
    "sessionid",
    # wsgi
    "x_csrftoken",
    "x_forwarded_for",
    "set_cookie",
    "cookie",
    "authorization",
    "x_api_key",
    # other common names used in the wild
    "aiohttp_session",  # aiohttp
    "connect.sid",  # Express
    "csrf_token",  # Pyramid
    "csrf",  # (this is a cookie name used in accepted answers on stack overflow)
    "_csrf",  # Express
    "_csrf_token",  # Bottle
    "PHPSESSID",  # PHP
    "_session",  # Sanic
    "symfony",  # Symfony
    "user_session",  # Vue
    "_xsrf",  # Tornado
    "XSRF-TOKEN",  # Angular, Laravel
]

DEFAULT_PII_DENYLIST = [
    "x_forwarded_for",
    "x_real_ip",
    "ip_address",
    "remote_addr",
]


class EventScrubber:
    def __init__(
        self, denylist=None, recursive=False, send_default_pii=False, pii_denylist=None
    ):
        # type: (Optional[List[str]], bool, bool, Optional[List[str]]) -> None
        """
        A scrubber that goes through the event payload and removes sensitive data configured through denylists.

        :param denylist: A security denylist that is always scrubbed, defaults to DEFAULT_DENYLIST.
        :param recursive: Whether to scrub the event payload recursively, default False.
        :param send_default_pii: Whether pii is sending is on, pii fields are not scrubbed.
        :param pii_denylist: The denylist to use for scrubbing when pii is not sent, defaults to DEFAULT_PII_DENYLIST.
        """
        self.denylist = DEFAULT_DENYLIST.copy() if denylist is None else denylist

        if not send_default_pii:
            pii_denylist = (
                DEFAULT_PII_DENYLIST.copy() if pii_denylist is None else pii_denylist
            )
            self.denylist += pii_denylist

        self.denylist = [x.lower() for x in self.denylist]
        self.recursive = recursive

    def scrub_list(self, lst):
        # type: (object) -> None
        """
        If a list is passed to this method, the method recursively searches the list and any
        nested lists for any dictionaries. The method calls scrub_dict on all dictionaries
        it finds.
        If the parameter passed to this method is not a list, the method does nothing.
        """
        if not isinstance(lst, list):
            return

        for v in lst:
            self.scrub_dict(v)  # no-op unless v is a dict
            self.scrub_list(v)  # no-op unless v is a list

    def scrub_dict(self, d):
        # type: (object) -> None
        """
        If a dictionary is passed to this method, the method scrubs the dictionary of any
        sensitive data. The method calls itself recursively on any nested dictionaries (
        including dictionaries nested in lists) if self.recursive is True.
        This method does nothing if the parameter passed to it is not a dictionary.
        """
        if not isinstance(d, dict):
            return

        for k, v in d.items():
            # The cast is needed because mypy is not smart enough to figure out that k must be a
            # string after the isinstance check.
            if isinstance(k, str) and k.lower() in self.denylist:
                d[k] = AnnotatedValue.substituted_because_contains_sensitive_data()
            elif self.recursive:
                self.scrub_dict(v)  # no-op unless v is a dict
                self.scrub_list(v)  # no-op unless v is a list

    def scrub_request(self, event):
        # type: (Event) -> None
        with capture_internal_exceptions():
            if "request" in event:
                if "headers" in event["request"]:
                    self.scrub_dict(event["request"]["headers"])
                if "cookies" in event["request"]:
                    self.scrub_dict(event["request"]["cookies"])
                if "data" in event["request"]:
                    self.scrub_dict(event["request"]["data"])

    def scrub_extra(self, event):
        # type: (Event) -> None
        with capture_internal_exceptions():
            if "extra" in event:
                self.scrub_dict(event["extra"])

    def scrub_user(self, event):
        # type: (Event) -> None
        with capture_internal_exceptions():
            if "user" in event:
                self.scrub_dict(event["user"])

    def scrub_breadcrumbs(self, event):
        # type: (Event) -> None
        with capture_internal_exceptions():
            if "breadcrumbs" in event:
                if "values" in event["breadcrumbs"]:
                    for value in event["breadcrumbs"]["values"]:
                        if "data" in value:
                            self.scrub_dict(value["data"])

    def scrub_frames(self, event):
        # type: (Event) -> None
        with capture_internal_exceptions():
            for frame in iter_event_frames(event):
                if "vars" in frame:
                    self.scrub_dict(frame["vars"])

    def scrub_spans(self, event):
        # type: (Event) -> None
        with capture_internal_exceptions():
            if "spans" in event:
                for span in cast(List[Dict[str, object]], event["spans"]):
                    if "data" in span:
                        self.scrub_dict(span["data"])

    def scrub_event(self, event):
        # type: (Event) -> None
        self.scrub_request(event)
        self.scrub_extra(event)
        self.scrub_user(event)
        self.scrub_breadcrumbs(event)
        self.scrub_frames(event)
        self.scrub_spans(event)
