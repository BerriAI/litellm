import gzip,json,time
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
root=Path('/verification')
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  path=self.path.split('?')[0]
  mode=json.loads((root/'spec_modes.json').read_text()).get(path,'ok')
  if mode=='timeout':time.sleep(35)
  code=404 if mode=='missing' else 401 if mode=='auth' else 200
  base=(root/'github-openapi.json').read_bytes()
  body=b'invalid JSON' if mode=='invalid' else base
  if mode=='large':body=json.dumps({**json.loads(base),'description':'x'*(12*1024*1024)}).encode()
  if mode=='compressed':body=gzip.compress(body)
  self.send_response(code);self.send_header('Content-Type','application/json')
  if mode=='compressed':self.send_header('Content-Encoding','gzip')
  if mode=='large':self.send_header('Content-Length',str(len(body)))
  self.end_headers()
  try:self.wfile.write(body)
  except (BrokenPipeError,ConnectionResetError):pass
  print(json.dumps({'path':path,'status':code,'response_bytes':len(body)}),flush=True)
 def do_POST(self):self.send_response(405);self.end_headers()
ThreadingHTTPServer(('0.0.0.0',8080),Handler).serve_forever()
