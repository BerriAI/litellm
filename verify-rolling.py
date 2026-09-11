import json,subprocess,os
from pathlib import Path
p=Path(__file__).parent
python='/Users/jvalluru/LiteLLM-work/litellm-7449/.venv/bin/python'
roots={'before':p/'before-source-final','after':Path('/Users/jvalluru/LiteLLM-work/litellm-7449')}
settings=dict(line.split('=',1) for line in (p/'.env').read_text().splitlines() if '=' in line)
script='''import asyncio,json,sys
from starlette.requests import Request
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import register_aggregate_client,open_gateway_dcr_client
async def main():
 d=json.load(sys.stdin)
 if 'open' in d:
  r=open_gateway_dcr_client(d['open']);print(json.dumps({'accepted':r is not None,'count':len(r.redirect_uris) if r else None}));return
 req=Request({'type':'http','method':'POST','scheme':'http','path':'/register','query_string':b'','headers':[(b'host',b'localhost')]})
 res=await register_aggregate_client(req,{'redirect_uris':[f'https://client.example/{i}' for i in range(d['count'])]})
 print(json.dumps({'status':res.status_code,'body':json.loads(res.body)}))
asyncio.run(main())
'''
def call(rev,data):
 env={**os.environ,'LITELLM_SALT_KEY':settings['LITELLM_SALT_KEY'],'PYTHONPATH':str(roots[rev])}
 r=subprocess.run([python,'-c',script],cwd=roots[rev],env=env,input=json.dumps(data),capture_output=True,text=True,check=True)
 return json.loads(r.stdout)
rows=[]
for minted in ('before','after'):
 for n in (1,3,4):
  c=call(minted,{'count':n});row={'minted_by':minted,'uri_count':n,'registration_status':c['status']}
  if c['status']==201:
   for accepted_by in ('before','after'):row['opened_by_'+accepted_by]=call(accepted_by,{'open':c['body']['client_id']})['accepted']
  rows.append(row)
result={'before':(p/'before-sha').read_text().strip(),'after':(p/'after-sha').read_text().strip(),'cases':rows}
(p/'rolling-compatibility.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
