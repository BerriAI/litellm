import json
from seed import request,root
sid=json.loads((root/'servers.json').read_text())['url']
status,result=request('POST','/mcp-rest/test/tools/list',{'server_id':sid,'url':'http://[::1]:9/mcp','transport':'http','auth_type':'basic'})
assert status==200 and not result.get('tools') and result.get('error')
evidence={'commit':(root/'after-sha').read_text().strip(),'url':'http://[::1]:9/mcp','status':status,'structuredConnectionError':bool(result.get('error')),'toolCount':len(result.get('tools',[]))}
(root/'ipv6-preview.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence))
