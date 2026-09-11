from pathlib import Path
import json,subprocess,sys,time
from api import root,request
label,sha=sys.argv[1:3]
servers=json.loads((root/'health-servers.json').read_text())
env=dict(x.split('=',1) for x in (root/'.env').read_text().splitlines())
records=[]
for case,server in servers.items():
 url='http://localhost:44896/v1/mcp/server/'+server['server_id']
 command=['curl','--silent','--show-error','--max-time','45','--config','-','--write-out','\n%{http_code}',url]
 started=time.monotonic()
 response=subprocess.check_output(command,input='header = "Authorization: Bearer '+env['LITELLM_MASTER_KEY']+'"\n',text=True)
 raw,status=response.rsplit('\n',1);body=json.loads(raw)
 record={'case':case,'method':'GET','url':url,'http_status':int(status),'health':body.get('status'),'error':body.get('health_check_error'),'elapsed_seconds':round(time.monotonic()-started,2)}
 records.append(record);print(json.dumps(record),flush=True)
(root/(label+'-health.json')).write_text(json.dumps({'commit':sha,'results':records},indent=2))
