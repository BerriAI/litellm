import asyncio
import contextvars
import gc
import http.server
import inspect
import threading
import weakref

native = globals()["native"]
started = threading.Event()
release = threading.Event()
server = None
url = None
context = contextvars.ContextVar("proof", default="unset")


def start_server():
    global server, url
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, self.headers.get_all("X-Proof"), body))
            if body == b"hold":
                started.set()
                release.wait(5)
            self.send_response(429)
            self.send_header("X-Reply", "one")
            self.send_header("X-Reply", "two")
            self.send_header("Content-Length", "3")
            self.end_headers()
            try:
                self.wfile.write(b"\x00\xffR")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:%s/ocr" % server.server_port
    return requests


requests = start_server()


class Graph(dict):
    pass


class Boundary:
    def __init__(self, *, asynchronous=False, nested=False, failure=None, hold=False):
        self.asynchronous = asynchronous
        self.nested = nested
        self.failure = failure
        self.hold = hold
        self.events = []
        self.thread = threading.get_ident()
        self.task = asyncio.current_task() if asynchronous else None
        self.result = object()
        self.error = LookupError("original callback error")

    def phase(self, name):
        assert threading.get_ident() == self.thread
        if self.asynchronous:
            assert asyncio.current_task() is self.task
            assert context.get() == ("initial" if name == "prepare" else "prepared")
        self.events.append(name)
        if self.failure == name:
            raise self.error
        if not self.nested and not self.hold:
            child = Boundary(nested=True)
            assert native.ocr_retained(child) is child.result
            assert child.events == ["prepare", "encode", "finish"]

    def prepare(self):
        self.phase("prepare")
        headers = Graph({"X-Proof": "original"})
        document = object()
        body = Graph(document=document, alias=document)
        body["cycle"] = body
        self.refs = (weakref.ref(headers), weakref.ref(body))
        self.view = {"headers": headers, "body": body}
        headers["X-Proof"] = "mutated"
        self.view["headers"] = {"replacement": True}
        self.view["body"] = {"replacement": True}
        return (headers, url, body, None)

    async def aprepare(self):
        await asyncio.sleep(0)
        roots = self.prepare()
        context.set("prepared")
        return roots

    def encode(self, roots):
        self.phase("encode")
        headers, target, body, files = roots
        assert headers is self.refs[0]() and body is self.refs[1]()
        assert headers["X-Proof"] == "mutated"
        assert body["document"] is body["alias"] and body["cycle"] is body
        assert files is None
        assert self.view == {"headers": {"replacement": True}, "body": {"replacement": True}}
        return (
            target,
            [(b"X-Proof", b"mutated"), (b"X-Proof", b"duplicate")],
            b"hold" if self.hold else b"\x00\xffQ",
            3.0,
        )

    def finish(self, wire):
        self.phase("finish")
        assert type(wire) is tuple and len(wire) == 3
        status, headers, content = wire
        assert status == 429
        assert type(headers) is list
        assert all(type(pair) is tuple and all(type(v) is bytes for v in pair) for pair in headers)
        assert [v for k, v in headers if k == b"x-reply"] == [b"one", b"two"]
        assert type(content) is bytes and content == b"\x00\xffR"
        assert all(ref() is not None for ref in self.refs)
        return self.result

    async def afinish(self, wire):
        await asyncio.sleep(0)
        return self.finish(wire)


def collected(boundary):
    gc.collect()
    assert all(ref() is None for ref in boundary.refs)
    assert boundary.view == {"headers": {"replacement": True}, "body": {"replacement": True}}


def check_error(boundary, error, phase):
    assert error is boundary.error
    names = []
    traceback = error.__traceback__
    while traceback:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert phase in names and "phase" in names
    assert boundary.events == ["prepare", "encode", "finish"][: ["prepare", "encode", "finish"].index(phase) + 1]


async def exercise():
    context.set("initial")
    boundary = Boundary(asynchronous=True)
    pending = native.aocr_retained(boundary)
    assert inspect.iscoroutine(pending)
    assert boundary.events == []
    assert await pending is boundary.result
    assert context.get() == "prepared"
    assert boundary.events == ["prepare", "encode", "finish"]
    collected(boundary)

    unused = Boundary(asynchronous=True)
    ref = weakref.ref(unused)
    pending = native.aocr_retained(unused)
    assert unused.events == []
    del unused
    assert ref() is not None
    pending.close()
    del pending
    gc.collect()
    assert ref() is None

    for phase in ("prepare", "encode", "finish"):
        context.set("initial")
        boundary = Boundary(asynchronous=True, nested=True, failure=phase)
        try:
            await native.aocr_retained(boundary)
        except LookupError as error:
            check_error(boundary, error, phase)
        else:
            raise AssertionError("callback error was swallowed")
        boundary.error.__traceback__ = None
        if phase != "prepare":
            collected(boundary)

    context.set("initial")
    boundary = Boundary(asynchronous=True, hold=True)

    async def cancellable():
        boundary.task = asyncio.current_task()
        await native.aocr_retained(boundary)

    task = asyncio.create_task(cancellable())
    assert await asyncio.to_thread(started.wait, 2)
    gc.collect()
    assert all(ref() is not None for ref in boundary.refs)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation was swallowed")
    assert boundary.events == ["prepare", "encode"]
    del task
    await asyncio.sleep(0)
    collected(boundary)
    release.set()


def run_ownership_contract():
    try:
        boundary = Boundary()
        assert native.ocr_retained(boundary) is boundary.result
        assert boundary.events == ["prepare", "encode", "finish"]
        collected(boundary)
        for phase in ("prepare", "encode", "finish"):
            boundary = Boundary(nested=True, failure=phase)
            try:
                native.ocr_retained(boundary)
            except LookupError as error:
                check_error(boundary, error, phase)
            else:
                raise AssertionError("callback error was swallowed")
            boundary.error.__traceback__ = None
            if phase != "prepare":
                collected(boundary)
        asyncio.run(asyncio.wait_for(exercise(), 15))
        assert requests
        assert all(
            path == "/ocr" and headers == ["mutated", "duplicate"] and body in (b"\x00\xffQ", b"hold")
            for path, headers, body in requests
        )
    finally:
        release.set()
        server.shutdown()
        server.server_close()


class TimeoutBoundary(Boundary):
    def __init__(self, *, timeout, url, asynchronous=False):
        super().__init__(asynchronous=asynchronous)
        self.timeout = timeout
        self.url = url

    def prepare(self):
        return ({}, self.url, {}, None)

    async def aprepare(self):
        return self.prepare()

    def encode(self, roots):
        headers, target, body, files = roots
        return (target, [], b"\x00", self.timeout)

    def finish(self, wire):
        raise AssertionError("client-side failure must not reach finish")


def run_error_contract():
    cases = [
        ("timeout", "http://10.255.255.1:9/", 0.05),
        ("refused", "http://127.0.0.1:1/", 1.0),
    ]
    for name, target, timeout in cases:
        boundary = TimeoutBoundary(timeout=timeout, url=target)
        try:
            native.ocr_retained(boundary)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{name} did not surface as RuntimeError")
