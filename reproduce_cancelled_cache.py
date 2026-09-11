import asyncio,json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import litellm
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager,MCPServer,MCPTransport,MCPAuth
class Handler(BaseHTTPRequestHandler):
 requests=0
 started=threading.Event()
 release=threading.Event()
 def do_GET(self):
  type(self).requests+=1
  if type(self).requests==1:
   self.started.set();self.release.wait(3)
  body=b'{"paths":{}}'
  self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
  try:self.wfile.write(body)
  except (BrokenPipeError,ConnectionResetError):pass
 def log_message(self,*args):pass
async def main():
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
 threading.Thread(target=server.serve_forever,daemon=True).start()
 litellm.user_url_allowed_hosts=['127.0.0.1:'+str(server.server_port)]
 manager=MCPServerManager()
 entry=MCPServer(server_id='cancelled',name='cancelled',transport=MCPTransport.http,auth_type=MCPAuth.none,spec_path='http://127.0.0.1:'+str(server.server_port)+'/spec.json')
 manager.registry={entry.server_id:entry}
 task=asyncio.create_task(manager.health_check_server(entry.server_id))
 assert await asyncio.to_thread(Handler.started.wait,2)
 task.cancel()
 first=await task
 Handler.release.set()
 second=await manager.health_check_server(entry.server_id)
 print(json.dumps({'requests':Handler.requests,'cancelled_status':first.status,'next_status':second.status,'next_error':second.health_check_error}))
 server.shutdown()
asyncio.run(main())
