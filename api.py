import json,urllib.request,urllib.error
from pathlib import Path
root=Path(__file__).parent
env=dict(x.split('=',1) for x in (root/'.env').read_text().splitlines())
def request(method,path,payload=None,key=None):
 req=urllib.request.Request('http://localhost:44896'+path,data=json.dumps(payload).encode() if payload is not None else None,headers={'Authorization':'Bearer '+(key or env['LITELLM_MASTER_KEY']),'Content-Type':'application/json'},method=method)
 try:
  with urllib.request.urlopen(req,timeout=60) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as e:return e.code,json.loads(e.read())
