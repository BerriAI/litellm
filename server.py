import json
import time
import queue
import socket
import uuid

channels = {}
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, status, body=b'', content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.startswith('/sse/'):
            parts = self.path.split('/')
            mode, phase = parts[2], parts[3]
            name = uuid.uuid4().hex
            messages = queue.Queue()
            channels[name] = (messages, mode, 'initialize' if phase == 'initialize' else 'tools/list')
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close')
            if mode == 'io-error':
                self.send_header('Content-Length', '1000000')
            self.end_headers()
            try:
                self.wfile.write(('event: endpoint\ndata: /messages/' + name + '\n\n').encode())
                self.wfile.flush()
                while True:
                    try:
                        message = messages.get(timeout=1)
                    except queue.Empty:
                        self.wfile.write(b': heartbeat\n\n')
                        self.wfile.flush()
                        continue
                    if message is None:
                        break
                    self.wfile.write(b'event: message\ndata: ' + message + b'\n\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                channels.pop(name, None)
                self.close_connection = True
            return
        self.send(200, b'<html>This is a diagnostic web page, not MCP</html>', 'text/html')

    def do_DELETE(self):
        self.send(200)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        if self.path.startswith('/messages/'):
            channel = channels.get(self.path.rsplit('/', 1)[1])
            if channel is None:
                self.send(404)
                return
            messages, mode, phase = channel
            self.send(202)
            if 'id' not in request or 'method' not in request:
                return
            if request['method'] == phase:
                if mode == 'bad-json':
                    messages.put(b'secret-invalid-json')
                    return
                if mode in ('closed', 'io-error'):
                    messages.put(None)
                    return
                if mode == 'silent':
                    return
            if request['method'] == 'initialize':
                result = {'protocolVersion': '2099-01-01' if mode == 'unknown' else request['params']['protocolVersion'], 'capabilities': {'tools':{}, 'logging':{}}, 'serverInfo': {'name':'sse-diagnostics','version':'1'}}
            elif request['method'] == 'tools/list':
                messages.put(json.dumps({'jsonrpc':'2.0','method':'notifications/message','params':{'level':'info','data':'Listing tools'}}).encode())
                messages.put(json.dumps({'jsonrpc':'2.0','id':'unmatched','result':{}}).encode())
                messages.put(json.dumps({'jsonrpc':'2.0','id':'server-ping','method':'ping'}).encode())
                result = {'tools':[{'name':'ping','inputSchema':{'type':'object'}}]}
            else:
                result = {'content':[{'type':'text','text':'pong'}],'isError':False}
            messages.put(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}).encode())
            return
        if self.path == '/empty-json':
            self.send(200, b'')
            return
        if self.path == '/empty-stream':
            self.send(200, b'', 'text/event-stream')
            return
        if self.path == '/io-error':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', '1000')
            self.end_headers()
            self.wfile.write(b'{"jsonrpc":')
            self.wfile.flush()
            self.close_connection = True
            return
        if self.path.startswith('/status/'):
            self.send(int(self.path.rsplit('/', 1)[1]), b'{"error":"diagnostic upstream failure"}')
            return
        if self.path == '/html':
            self.send(200, b'<html>Diagnostic web page</html>', 'text/html')
            return
        if self.path == '/bad-json':
            self.send(200, b'not valid JSON')
            return
        if self.path == '/bad-rpc':
            self.send(200, b'{"unexpected":"not a JSON-RPC response"}')
            return
        if self.path == '/slow':
            time.sleep(40)
        if 'id' not in request:
            self.send(202)
            return
        if self.path == '/rpc-error':
            self.send(200, json.dumps({'jsonrpc':'2.0','id':request['id'],'error':{'code':-32603,'message':'diagnostic-private-detail'}}).encode())
            return
        if self.path == '/bad-schema':
            result = {'unexpected':'not a valid MCP result'}
        elif request['method'] == 'initialize':
            result = {'protocolVersion': '2099-01-01' if self.path == '/unknown' else request['params']['protocolVersion'],
                      'capabilities': {'tools':{}}, 'serverInfo': {'name':'local-diagnostics','version':'1.0'}}
        elif request['method'] == 'tools/list':
            result = {'tools':[{'name':'diagnostic_ping','description':'Returns a fixed diagnostic response','inputSchema':{'type':'object','properties':{}}}]}
        elif request['method'] == 'tools/call':
            result = {'content':[{'type':'text','text':'diagnostic ok'}],'isError':False}
        else:
            result = {}
        self.send(200, json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}).encode())


ThreadingHTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
