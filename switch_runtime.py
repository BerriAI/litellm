from pathlib import Path
import re,subprocess,sys,time,urllib.request
root=Path(__file__).parent
sha=sys.argv[1]
(root/'spec_modes.json').write_text('{}')
p=root/'compose.yaml';p.write_text(re.sub(r'image: litellm-4896:[a-f0-9]+', 'image: litellm-4896:'+sha[:10],p.read_text(),count=1))
subprocess.run(['docker','compose','up','-d','--force-recreate','gateway'],cwd=root,check=True)
for _ in range(45):
 try:
  with urllib.request.urlopen('http://localhost:44896/health/readiness',timeout=2) as r:
   if r.status==200:break
 except Exception:time.sleep(1)
else:raise RuntimeError('Gateway not ready')
(root/'spec_modes.json').write_text('{"/missing.json":"missing","/invalid.json":"invalid","/slow.json":"timeout","/large.json":"large","/compressed.json":"compressed"}')
print('Gateway ready for '+sha)
