from litellm.proxy.rust_forward.server import Listener, PythonServer, Server


def python_server(
    server: Server = Server.UVICORN, listener: Listener = Listener.TCP, trusted_proxies: str = "127.0.0.1"
) -> PythonServer:
    return PythonServer(server, listener, "0.0.0.0", 4000, 2, 2.5, trusted_proxies)
