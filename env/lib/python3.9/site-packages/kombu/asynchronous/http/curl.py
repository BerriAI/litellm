"""HTTP Client using pyCurl."""

from __future__ import annotations

from collections import deque
from functools import partial
from io import BytesIO
from time import time

from kombu.asynchronous.hub import READ, WRITE, Hub, get_event_loop
from kombu.exceptions import HttpError
from kombu.utils.encoding import bytes_to_str

from .base import BaseClient

try:
    import pycurl
except ImportError:  # pragma: no cover
    pycurl = Curl = METH_TO_CURL = None
else:
    from pycurl import Curl

    METH_TO_CURL = {
        'GET': pycurl.HTTPGET,
        'POST': pycurl.POST,
        'PUT': pycurl.UPLOAD,
        'HEAD': pycurl.NOBODY,
    }

__all__ = ('CurlClient',)

DEFAULT_USER_AGENT = 'Mozilla/5.0 (compatible; pycurl)'
EXTRA_METHODS = frozenset(['DELETE', 'OPTIONS', 'PATCH'])


class CurlClient(BaseClient):
    """Curl HTTP Client."""

    Curl = Curl

    def __init__(self, hub: Hub | None = None, max_clients: int = 10):
        if pycurl is None:
            raise ImportError('The curl client requires the pycurl library.')
        hub = hub or get_event_loop()
        super().__init__(hub)
        self.max_clients = max_clients

        self._multi = pycurl.CurlMulti()
        self._multi.setopt(pycurl.M_TIMERFUNCTION, self._set_timeout)
        self._multi.setopt(pycurl.M_SOCKETFUNCTION, self._handle_socket)
        self._curls = [self.Curl() for i in range(max_clients)]
        self._free_list = self._curls[:]
        self._pending = deque()
        self._fds = {}

        self._socket_action = self._multi.socket_action
        self._timeout_check_tref = self.hub.call_repeatedly(
            1.0, self._timeout_check,
        )

        # pycurl 7.29.0 workaround
        dummy_curl_handle = pycurl.Curl()
        self._multi.add_handle(dummy_curl_handle)
        self._multi.remove_handle(dummy_curl_handle)

    def close(self):
        self._timeout_check_tref.cancel()
        for _curl in self._curls:
            _curl.close()
        self._multi.close()

    def add_request(self, request):
        self._pending.append(request)
        self._process_queue()
        self._set_timeout(0)
        return request

    # the next two methods are used for linux/epoll workaround:
    # we temporarily remove all curl fds from hub, so curl cannot
    # close a fd which is still inside epoll
    def _pop_from_hub(self):
        for fd in self._fds:
            self.hub.remove(fd)

    def _push_to_hub(self):
        for fd, events in self._fds.items():
            if events & READ:
                self.hub.add_reader(fd, self.on_readable, fd)
            if events & WRITE:
                self.hub.add_writer(fd, self.on_writable, fd)

    def _handle_socket(self, event, fd, multi, data, _pycurl=pycurl):
        if event == _pycurl.POLL_REMOVE:
            if fd in self._fds:
                self._fds.pop(fd, None)
        else:
            if event == _pycurl.POLL_IN:
                self._fds[fd] = READ
            elif event == _pycurl.POLL_OUT:
                self._fds[fd] = WRITE
            elif event == _pycurl.POLL_INOUT:
                self._fds[fd] = READ | WRITE

    def _set_timeout(self, msecs):
        self.hub.call_later(msecs, self._timeout_check)

    def _timeout_check(self, _pycurl=pycurl):
        self._pop_from_hub()
        try:
            while 1:
                try:
                    ret, _ = self._multi.socket_all()
                except pycurl.error as exc:
                    ret = exc.args[0]
                if ret != _pycurl.E_CALL_MULTI_PERFORM:
                    break
        finally:
            self._push_to_hub()
        self._process_pending_requests()

    def on_readable(self, fd, _pycurl=pycurl):
        return self._on_event(fd, _pycurl.CSELECT_IN)

    def on_writable(self, fd, _pycurl=pycurl):
        return self._on_event(fd, _pycurl.CSELECT_OUT)

    def _on_event(self, fd, event, _pycurl=pycurl):
        self._pop_from_hub()
        try:
            while 1:
                try:
                    ret, _ = self._socket_action(fd, event)
                except pycurl.error as exc:
                    ret = exc.args[0]
                if ret != _pycurl.E_CALL_MULTI_PERFORM:
                    break
        finally:
            self._push_to_hub()
        self._process_pending_requests()

    def _process_pending_requests(self):
        while 1:
            q, succeeded, failed = self._multi.info_read()
            for curl in succeeded:
                self._process(curl)
            for curl, errno, reason in failed:
                self._process(curl, errno, reason)
            if q == 0:
                break
        self._process_queue()

    def _process_queue(self):
        while 1:
            started = 0
            while self._free_list and self._pending:
                started += 1
                curl = self._free_list.pop()
                request = self._pending.popleft()
                headers = self.Headers()
                buf = BytesIO()
                curl.info = {
                    'headers': headers,
                    'buffer': buf,
                    'request': request,
                    'curl_start_time': time(),
                }
                self._setup_request(curl, request, buf, headers)
                self._multi.add_handle(curl)
            if not started:
                break

    def _process(self, curl, errno=None, reason=None, _pycurl=pycurl):
        info, curl.info = curl.info, None
        self._multi.remove_handle(curl)
        self._free_list.append(curl)
        buffer = info['buffer']
        if errno:
            code = 599
            error = HttpError(code, reason)
            error.errno = errno
            effective_url = None
            buffer.close()
            buffer = None
        else:
            error = None
            code = curl.getinfo(_pycurl.HTTP_CODE)
            effective_url = curl.getinfo(_pycurl.EFFECTIVE_URL)
            buffer.seek(0)
        # try:
        request = info['request']
        request.on_ready(self.Response(
            request=request, code=code, headers=info['headers'],
            buffer=buffer, effective_url=effective_url, error=error,
        ))

    def _setup_request(self, curl, request, buffer, headers, _pycurl=pycurl):
        setopt = curl.setopt
        setopt(_pycurl.URL, bytes_to_str(request.url))

        # see tornado curl client
        request.headers.setdefault('Expect', '')
        request.headers.setdefault('Pragma', '')

        setopt(
            _pycurl.HTTPHEADER,
            ['{}: {}'.format(*h) for h in request.headers.items()],
        )

        setopt(
            _pycurl.HEADERFUNCTION,
            partial(request.on_header or self.on_header, request.headers),
        )
        setopt(
            _pycurl.WRITEFUNCTION, request.on_stream or buffer.write,
        )
        setopt(
            _pycurl.FOLLOWLOCATION, request.follow_redirects,
        )
        setopt(
            _pycurl.USERAGENT,
            bytes_to_str(request.user_agent or DEFAULT_USER_AGENT),
        )
        if request.network_interface:
            setopt(_pycurl.INTERFACE, request.network_interface)
        setopt(
            _pycurl.ENCODING, 'gzip,deflate' if request.use_gzip else 'none',
        )
        if request.proxy_host:
            if not request.proxy_port:
                raise ValueError('Request with proxy_host but no proxy_port')
            setopt(_pycurl.PROXY, request.proxy_host)
            setopt(_pycurl.PROXYPORT, request.proxy_port)
            if request.proxy_username:
                setopt(_pycurl.PROXYUSERPWD, '{}:{}'.format(
                    request.proxy_username, request.proxy_password or ''))

        setopt(_pycurl.SSL_VERIFYPEER, 1 if request.validate_cert else 0)
        setopt(_pycurl.SSL_VERIFYHOST, 2 if request.validate_cert else 0)
        if request.ca_certs is not None:
            setopt(_pycurl.CAINFO, request.ca_certs)

        setopt(_pycurl.IPRESOLVE, pycurl.IPRESOLVE_WHATEVER)

        for meth in METH_TO_CURL.values():
            setopt(meth, False)
        try:
            meth = METH_TO_CURL[request.method]
        except KeyError:
            curl.setopt(_pycurl.CUSTOMREQUEST, request.method)
        else:
            curl.unsetopt(_pycurl.CUSTOMREQUEST)
            setopt(meth, True)

        if request.method in ('POST', 'PUT'):
            body = request.body.encode('utf-8') if request.body else b''
            reqbuffer = BytesIO(body)
            setopt(_pycurl.READFUNCTION, reqbuffer.read)
            if request.method == 'POST':

                def ioctl(cmd):
                    if cmd == _pycurl.IOCMD_RESTARTREAD:
                        reqbuffer.seek(0)
                setopt(_pycurl.IOCTLFUNCTION, ioctl)
                setopt(_pycurl.POSTFIELDSIZE, len(body))
            else:
                setopt(_pycurl.INFILESIZE, len(body))
        elif request.method == 'GET':
            assert not request.body

        if request.auth_username is not None:
            auth_mode = {
                'basic': _pycurl.HTTPAUTH_BASIC,
                'digest': _pycurl.HTTPAUTH_DIGEST
            }[request.auth_mode or 'basic']
            setopt(_pycurl.HTTPAUTH, auth_mode)
            userpwd = '{}:{}'.format(
                request.auth_username, request.auth_password or '',
            )
            setopt(_pycurl.USERPWD, userpwd)
        else:
            curl.unsetopt(_pycurl.USERPWD)

        if request.client_cert is not None:
            setopt(_pycurl.SSLCERT, request.client_cert)
        if request.client_key is not None:
            setopt(_pycurl.SSLKEY, request.client_key)

        if request.on_prepare is not None:
            request.on_prepare(curl)
