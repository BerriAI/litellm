import asyncio,json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import litellm
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServer,MCPTransport,MCPAuth
class Handler(BaseHTTPRequestHandler):
 requests=0
 sent=0
 def do_GET(self):
  type(self).requests+=1
  body=json.dumps({'openapi':'3.0.0','paths':{},'description':'x'*(12*1024*1024)}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
  try:self.wfile.write(body);type(self).sent+=len(body)
  except (BrokenPipeError,ConnectionResetError):pass
 def log_message(self,*args):pass
async def main():
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
 threading.Thread(target=server.serve_forever,daemon=True).start()
 litellm.user_url_allowed_hosts=['127.0.0.1:'+str(server.server_port)]
 manager=MCPServerManager()
 entry=MCPServer(server_id='large',name='large',transport=MCPTransport.http,auth_type=MCPAuth.none,spec_path='http://127.0.0.1:'+str(server.server_port)+'/spec.json')
 manager.registry={entry.server_id:entry}
 results=await asyncio.gather(*(manager.health_check_server('large') for _ in range(3)))
 print(json.dumps({'requests':Handler.requests,'bytes_sent':Handler.sent,'statuses':[r.status for r in results],'errors':[r.health_check_error for r in results]}))
 server.shutdown()
asyncio.run(main())
