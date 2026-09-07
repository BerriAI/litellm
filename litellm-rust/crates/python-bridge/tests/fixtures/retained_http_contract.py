import asyncio
import contextvars
import gc
import http.client
import http.server
import inspect
import sys
import threading
import weakref
from copy import deepcopy
from typing import NamedTuple

native = globals()["native"]
started = threading.Event()
release = threading.Event()
server = None
server_thread = None
url = None
context = contextvars.ContextVar("proof", default="unset")


def start_server():
    global server, server_thread, url
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
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = "http://127.0.0.1:%s/ocr" % server.server_port
    return requests


requests = start_server()


def stop_server():
    release.set()
    server.shutdown()
    server.server_close()
    server_thread.join(5)
    assert not server_thread.is_alive()


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


def run_cold_cache_reentry(filename):
    """UC-COLD-REENTRY: cold compilation permits nested native calls and subsequent transport."""
    events = []
    error = LookupError("cold-cache preparation error")

    class FailingBoundary:
        async def aprepare(self):
            raise error

    def invoke_failure():
        pending = native.aocr_retained(FailingBoundary())
        observed = None
        try:
            pending.send(None)
        except LookupError as caught:
            observed = caught
        finally:
            pending.close()
            error.__traceback__ = None
        assert observed is error

    def audit(event, args):
        if event == "compile" and args[1] == filename and not events:
            events.append("entered")
            print(f"UC-COLD-REENTRY: entering {filename}", flush=True)  # noqa: T201  # diagnose a deadlocked child process
            invoke_failure()
            events.append("nested completed")

    async def successful_async():
        context.set("initial")
        boundary = Boundary(asynchronous=True, nested=True)
        assert await native.aocr_retained(boundary) is boundary.result
        assert boundary.events == ["prepare", "encode", "finish"]
        collected(boundary)

    try:
        if filename == "retained_callback.py":
            native.aocr_retained(object()).close()
        sys.addaudithook(audit)
        invoke_failure()
        events.append("outer completed")
        assert events == ["entered", "nested completed", "outer completed"]
        assert requests == []
        boundary = Boundary(nested=True)
        assert native.ocr_retained(boundary) is boundary.result
        assert boundary.events == ["prepare", "encode", "finish"]
        collected(boundary)
        asyncio.run(successful_async())
        assert len(requests) == 2
    finally:
        stop_server()


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
        boundary = Boundary(hold=True)
        observations = []

        def observe_blocked_transport():
            received = started.wait(2)
            gc.collect()
            observations.append((received, tuple(ref() is not None for ref in boundary.refs)))
            release.set()

        observer = threading.Thread(target=observe_blocked_transport)
        observer.start()
        try:
            assert native.ocr_retained(boundary) is boundary.result
        finally:
            release.set()
            observer.join(5)
        assert not observer.is_alive()
        assert observations == [(True, (True, True))]
        assert boundary.events == ["prepare", "encode", "finish"]
        collected(boundary)
        started.clear()
        release.clear()
        asyncio.run(asyncio.wait_for(exercise(), 15))
        assert requests
        assert all(
            path == "/ocr" and headers == ["mutated", "duplicate"] and body in (b"\x00\xffQ", b"hold")
            for path, headers, body in requests
        )
    finally:
        stop_server()


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
        return (target, [], b"hold", self.timeout)

    def finish(self, wire):
        raise AssertionError("client-side failure must not reach finish")


def run_error_contract():
    cases = [
        ("timeout", url, 0.05),
        ("refused", "http://127.0.0.1:1/", 1.0),
    ]
    try:
        for name, target, timeout in cases:
            boundary = TimeoutBoundary(timeout=timeout, url=target)
            try:
                native.ocr_retained(boundary)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"{name} did not surface as RuntimeError")
    finally:
        stop_server()


class FrozenObservation(NamedTuple):
    root_tuple_identity: bool
    roots_identity: tuple[bool, bool]
    document_is_caller: bool
    document_alias: bool
    logging_envelope_identity: bool
    logging_roots_identity: tuple[bool, bool]
    logging_fields: tuple[str, str]
    retained_fields: tuple[str, str]
    caller_value: str
    caught_error_observation: tuple[str, ...]
    callback_count: int
    phases: tuple[str, ...]
    request: tuple[str, tuple[str, ...], bytes]
    response: tuple[int, tuple[bytes, ...], bytes]


class ContractCallback:
    def __init__(self, scenario, caller):
        self.scenario = scenario
        self.caller = caller
        self.mutate_caller = lambda: caller.__setitem__("value", "closure")
        self.calls = 0
        self.error = LookupError("caught callback failure")
        self.writer = None
        self.write = threading.Event()
        self.written = threading.Event()

    def __call__(self, view):
        self.calls += 1
        self.view = view
        self.headers, self.body = view["headers"], view["body"]
        if self.scenario == "caller_closure":
            self.mutate_caller()
        elif self.scenario == "field_replace":
            document = Graph(value="replacement")
            view["headers"] = Graph({"X-Proof": "replacement"})
            view["body"] = Graph(document=document, alias=document)
        elif self.scenario == "delayed_after_prepare_before_encode_read":

            def writer():
                if self.write.wait(5):
                    self.mutate("delayed")
                    self.written.set()

            self.writer = threading.Thread(target=writer)
            self.writer.start()
        else:
            assert self.scenario in ("retain_mutate", "mutate_then_caught_error_observe")
            self.mutate("mutated" if self.scenario == "retain_mutate" else "caught")
            if self.scenario == "mutate_then_caught_error_observe":
                raise self.error

    def mutate(self, value):
        self.headers["X-Proof"] = value
        self.body["document"]["value"] = value

    def before_encode_read(self):
        if self.writer is not None:
            self.write.set()
            assert self.written.wait(2), "scheduled mutation did not complete before encoder read"
            self.writer.join(2)
            assert not self.writer.is_alive()

    def close(self):
        self.write.set()
        if self.writer is not None:
            self.writer.join(5)
            assert not self.writer.is_alive()
        self.error.__traceback__ = None


class TransportContractBinding:
    def __init__(self, caller, callback):
        self.caller = caller
        self.callback = callback
        self.phases = []
        self.caught = ()

    def prepare(self):
        self.phases.append("prepare")
        headers = Graph({"X-Proof": "original"})
        body = Graph(document=self.caller, alias=self.caller)
        self.view = {"headers": headers, "body": body}
        self.prepared_roots = (headers, url, body, None)
        try:
            self.callback(self.view)
        except LookupError as error:
            if error is not self.callback.error:
                raise
            self.caught = (str(error), headers["X-Proof"], body["document"]["value"])
        return self.prepared_roots

    async def aprepare(self):
        await asyncio.sleep(0)
        return self.prepare()

    def encode(self, roots):
        self.callback.before_encode_read()
        self.phases.append("encode")
        headers, target, body, files = roots
        assert files is None
        self.root_tuple_identity = roots is self.prepared_roots
        self.roots_identity = (headers is self.callback.headers, body is self.callback.body)
        self.document_is_caller = body["document"] is self.callback.caller
        self.document_alias = body["document"] is body["alias"]
        return (
            target,
            [(b"X-Proof", headers["X-Proof"].encode())],
            b"\x00" + body["document"]["value"].encode() + b"\xff",
            3.0,
        )

    def finish(self, wire):
        self.phases.append("finish")
        status, headers, content = wire
        return (status, tuple(value for key, value in headers if key == b"x-reply"), content)

    async def afinish(self, wire):
        await asyncio.sleep(0)
        return self.finish(wire)


class BindingVariant:
    def __init__(self, binding, variant):
        self.binding = binding
        self.variant = variant

    def before_prepare(self):
        if self.variant == "copy_caller_inputs_before_prepare":
            self.binding.caller = deepcopy(self.binding.caller)

    def at_prepare_return(self, roots):
        if self.variant == "copy_roots_after_prepare":
            return deepcopy(roots)
        if self.variant == "reconstruct_root_tuple":
            return tuple(list(roots))
        if self.variant == "reconstruct_logging_envelope":
            self.binding.view = dict(self.binding.view)
        return roots

    def prepare(self):
        self.before_prepare()
        return self.at_prepare_return(self.binding.prepare())

    async def aprepare(self):
        self.before_prepare()
        return self.at_prepare_return(await self.binding.aprepare())

    def encode(self, roots):
        if self.variant == "encode_logging_replacements":
            headers, target, body, files = roots
            return self.binding.encode((self.binding.view["headers"], target, self.binding.view["body"], files))
        return self.binding.encode(roots)

    def finish(self, wire):
        return self.binding.finish(wire)

    async def afinish(self, wire):
        return await self.binding.afinish(wire)


def reference_post(encoded):
    target, headers, body, timeout = encoded
    assert target == url
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=timeout)
    try:
        connection.putrequest("POST", "/ocr")
        for key, value in headers:
            connection.putheader(key.decode("ascii"), value.decode("ascii"))
        connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        return (
            response.status,
            [(key.lower().encode(), value.encode()) for key, value in response.getheaders()],
            response.read(),
        )
    finally:
        connection.close()


async def reference_async(binding):
    roots = await binding.aprepare()
    wire = await asyncio.to_thread(reference_post, binding.encode(roots))
    return await binding.afinish(wire)


def original_observation(value, *, header=None, logging=None, caught=()):
    fields = (header or value, value)
    return FrozenObservation(
        True,
        (True, True),
        True,
        True,
        True,
        (False, False) if logging else (True, True),
        logging or fields,
        fields,
        value,
        caught,
        1,
        ("prepare", "encode", "finish"),
        ("/ocr", (fields[0],), b"\x00" + value.encode() + b"\xff"),
        (429, (b"one", b"two"), b"\x00\xffR"),
    )


ORIGINAL_OBSERVATIONS = {
    "retain_mutate": original_observation("mutated"),
    "caller_closure": original_observation("closure", header="original"),
    "delayed_after_prepare_before_encode_read": original_observation("delayed"),
    "field_replace": original_observation("original", logging=("replacement", "replacement")),
    "mutate_then_caught_error_observe": original_observation(
        "caught", caught=("caught callback failure", "caught", "caught")
    ),
}

VARIANT_OBSERVATIONS = {
    ("retain_mutate", "copy_caller_inputs_before_prepare"): {
        "caller_value": "original",
        "document_is_caller": False,
    },
    ("caller_closure", "copy_caller_inputs_before_prepare"): {
        "document_is_caller": False,
        "logging_fields": ("original", "original"),
        "retained_fields": ("original", "original"),
        "request": ("/ocr", ("original",), b"\x00original\xff"),
    },
    ("retain_mutate", "copy_roots_after_prepare"): {
        "root_tuple_identity": False,
        "roots_identity": (False, False),
        "document_is_caller": False,
    },
    ("delayed_after_prepare_before_encode_read", "copy_roots_after_prepare"): {
        "root_tuple_identity": False,
        "roots_identity": (False, False),
        "document_is_caller": False,
        "request": ("/ocr", ("original",), b"\x00original\xff"),
    },
    ("mutate_then_caught_error_observe", "copy_roots_after_prepare"): {
        "root_tuple_identity": False,
        "roots_identity": (False, False),
        "document_is_caller": False,
    },
    ("field_replace", "encode_logging_replacements"): {
        "root_tuple_identity": False,
        "roots_identity": (False, False),
        "document_is_caller": False,
        "request": ("/ocr", ("replacement",), b"\x00replacement\xff"),
    },
    ("retain_mutate", "reconstruct_logging_envelope"): {"logging_envelope_identity": False},
    ("field_replace", "reconstruct_logging_envelope"): {"logging_envelope_identity": False},
    ("retain_mutate", "reconstruct_root_tuple"): {"root_tuple_identity": False},
    ("delayed_after_prepare_before_encode_read", "reconstruct_root_tuple"): {"root_tuple_identity": False},
    ("field_replace", "reconstruct_root_tuple"): {"root_tuple_identity": False},
}


def run_comparison_contract(scenario, variant):
    try:
        expected = ORIGINAL_OBSERVATIONS[scenario]
        if variant != "original":
            expected = expected._replace(**VARIANT_OBSERVATIONS[(scenario, variant)])
        for mode in ("reference-sync", "reference-async", "native-sync", "native-async"):
            caller = Graph(value="original")
            callback = ContractCallback(scenario, caller)
            binding = TransportContractBinding(caller, callback)
            selected = binding if variant == "original" else BindingVariant(binding, variant)
            offset = len(requests)
            try:
                if mode == "reference-sync":
                    roots = selected.prepare()
                    response = selected.finish(reference_post(selected.encode(roots)))
                elif mode == "reference-async":
                    response = asyncio.run(reference_async(selected))
                elif mode == "native-sync":
                    response = native.ocr_retained(selected)
                else:
                    response = asyncio.run(native.aocr_retained(selected))
                assert len(requests) == offset + 1, (scenario, variant, mode, requests[offset:])
                path, headers, body = requests[offset]
                observed = FrozenObservation(
                    binding.root_tuple_identity,
                    binding.roots_identity,
                    binding.document_is_caller,
                    binding.document_alias,
                    binding.view is callback.view,
                    (binding.view["headers"] is callback.headers, binding.view["body"] is callback.body),
                    (binding.view["headers"]["X-Proof"], binding.view["body"]["document"]["value"]),
                    (callback.headers["X-Proof"], callback.body["document"]["value"]),
                    caller["value"],
                    binding.caught,
                    callback.calls,
                    tuple(binding.phases),
                    (path, tuple(headers), body),
                    response,
                )
                assert observed == expected, (scenario, variant, mode, observed, expected)
            finally:
                callback.close()
    finally:
        stop_server()
