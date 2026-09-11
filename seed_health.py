from api import request,root
import json
(root/'spec_modes.json').write_text('{}')
existing=json.loads((root/'servers.json').read_text())
out={}
for name,auth,path in [
 ('remote_none','none','https://raw.githubusercontent.com/BerriAI/litellm/78cadacc6251c2683869cca7b2b7a409e047adab/github-openapi.json'),
 ('remote_bearer','bearer_token','https://raw.githubusercontent.com/BerriAI/litellm/78cadacc6251c2683869cca7b2b7a409e047adab/github-openapi.json'),
 ('remote_missing','none','http://spec-host:8080/missing.json'),
 ('remote_invalid','none','http://spec-host:8080/invalid.json'),
 ('remote_timeout','none','http://spec-host:8080/slow.json'),
]:
 payload={'server_name':name,'alias':name,'transport':'http','url':'https://api.github.com','spec_path':path,'auth_type':auth,'is_byok':True,'allow_all_keys':True,'mcp_info':{'server_name':name},'static_headers':{'User-Agent':'LiteLLM-LIT4896-readonly-repro'}}
 status,body=request('POST','/v1/mcp/server',payload)
 assert status==201,(status,body)
 out[name]={'server_id':body['server_id'],'create_status':status}
status,body=request('POST','/v1/mcp/server',{'server_name':'native_http','alias':'native_http','transport':'http','url':'https://mcp.deepwiki.com/mcp','auth_type':'none','allow_all_keys':True,'mcp_info':{'server_name':'native_http'}})
assert status==201,(status,body)
out['native_http']={'server_id':body['server_id'],'create_status':status}
out['local_none']=existing['byok_default'];out['local_bearer']=existing['byok_bearer']
(root/'health-servers.json').write_text(json.dumps(out,indent=2))
(root/'spec_modes.json').write_text(json.dumps({'/missing.json':'missing','/invalid.json':'invalid','/slow.json':'timeout'}))
print(json.dumps(out))
