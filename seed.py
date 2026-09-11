from api import request,root
import json
out={}
for alias,auth_type in [('byok_default','none'),('byok_bearer','bearer_token')]:
 payload={'server_name':alias,'alias':alias,'transport':'http','url':'https://api.github.com','spec_path':'/verification/github-openapi.json','auth_type':auth_type,'is_byok':True,'allow_all_keys':True,'mcp_info':{'server_name':alias},'static_headers':{'User-Agent':'LiteLLM-LIT4896-readonly-repro'}}
 status,body=request('POST','/v1/mcp/server',payload)
 out[alias]={'status':status,'server_id':body.get('server_id')}
 if status!=200 and status!=201: print(json.dumps(body)[:500])
print(json.dumps(out));(root/'servers.json').write_text(json.dumps(out,indent=2))
