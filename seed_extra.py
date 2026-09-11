from api import request,root
import json
(root/'spec_modes.json').write_text('{}')
servers=json.loads((root/'health-servers.json').read_text())
for name,path in [('remote_large','large.json'),('remote_compressed','compressed.json')]:
 if name in servers:continue
 status,body=request('POST','/v1/mcp/server',{'server_name':name,'alias':name,'transport':'http','url':'https://api.github.com','spec_path':'http://spec-host:8080/'+path,'auth_type':'none','is_byok':True,'allow_all_keys':True,'mcp_info':{'server_name':name}})
 assert status==201,(status,body)
 servers[name]={'server_id':body['server_id'],'create_status':status}
(root/'health-servers.json').write_text(json.dumps(servers,indent=2))
(root/'spec_modes.json').write_text('{"/missing.json":"missing","/invalid.json":"invalid","/slow.json":"timeout","/large.json":"large","/compressed.json":"compressed"}')
print('Registered size and compression scenarios')
