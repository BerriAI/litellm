import asyncio, base64, hashlib, json, secrets, sys
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs
from playwright.async_api import async_playwright

root=Path(__file__).parent
label=sys.argv[1]
base='http://127.0.0.1:47449'
out=root/label
out.mkdir(exist_ok=True)
settings=dict(line.split('=',1) for line in (root/'.env').read_text().splitlines() if '=' in line)
sha=(root/f'{label}-sha').read_text().strip()
uris=['https://insiders.vscode.dev/redirect','https://vscode.dev/redirect','http://127.0.0.1/','http://127.0.0.1:33418/','http://127.0.0.1:33419/']

def rpc(body):
 return json.loads(next((line[6:] for line in body.splitlines() if line.startswith('data: ')),body))

async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(headless=True,executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
  context=await browser.new_context(viewport={'width':1440,'height':1100})
  api=context.request
  results={'commit':sha,'base':base,'registration':[],'flows':[]}
  for n in (2,3,4,5):
   payload=json.loads((root/f'register-{n}.request.json').read_text())
   response=await api.post(base+'/register',data=payload)
   body=await response.json()
   results['registration'].append({'count':n,'status':response.status,'request':payload,'response':{k:v for k,v in body.items() if k!='client_id'},'client_id_bytes':len(body.get('client_id','').encode())})
   assert response.status==(201 if n<=(3 if label=='before' else 4) else 400)
  for case,callbacks in [('existing-three',uris[:3]),('vscode-four',uris[:4]),('max-length-four',[f'https://client.example/{i}/'.ljust(256,'a') for i in range(4)])]:
   response=await api.post(base+'/register',data={'redirect_uris':callbacks,'token_endpoint_auth_method':'none'})
   rec={'case':case,'registration_status':response.status};results['flows'].append(rec)
   if response.status!=201:
    rec['error']=await response.json();continue
   client_id=(await response.json())['client_id']
   login=await api.post(base+'/login',form={'username':'admin','password':settings['LITELLM_MASTER_KEY']},max_redirects=0)
   assert login.status==303,(login.status,await login.text())
   verifier=secrets.token_urlsafe(48)
   challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
   params={'client_id':client_id,'redirect_uri':callbacks[-1],'response_type':'code','state':'lit7449-state','code_challenge':challenge,'code_challenge_method':'S256','resource':base+'/mcp'}
   response=await api.get(base+'/authorize?'+urlencode(params),max_redirects=0)
   assert response.status==303,(response.status,await response.text())
   location=response.headers['location']; handle=parse_qs(urlparse(location).query)['connect_flow'][0]
   page=await context.new_page()
   await page.goto(location)
   await page.wait_for_timeout(2000)
   await page.screenshot(path=str(out/f'{case}-consent.png'),full_page=True)
   (out/f'{case}-consent-text.txt').write_text(await page.locator('body').inner_text())
   cookies=await context.cookies()
   rec['flow_cookie_stored']=any(c['name']=='mcp_connect_flow_'+handle for c in cookies)
   assert rec['flow_cookie_stored']
   rec['cookie_names']=[c['name'] for c in cookies]
   response=await api.post(base+'/authorize/complete',form={'flow':handle,'decision':'approve'},max_redirects=0)
   assert response.status==303,(response.status,await response.text())
   query=parse_qs(urlparse(response.headers['location']).query)
   assert query['state']==['lit7449-state']
   code=query['code'][0]
   form={'grant_type':'authorization_code','client_id':client_id,'code':code,'redirect_uri':callbacks[-1],'code_verifier':verifier,'resource':base+'/mcp'}
   wrong=await api.post(base+'/token',form={**form,'code_verifier':'x'*64})
   assert wrong.status==400;rec['wrong_pkce_status']=wrong.status
   response=await api.post(base+'/token',form=form)
   assert response.status==200,(response.status,await response.text())
   tokens=await response.json();rec['token_status']=response.status
   rec['access_token_bytes']=len(tokens['access_token'].encode());rec['refresh_token_bytes']=len(tokens['refresh_token'].encode());rec['client_id_bytes']=len(client_id.encode());rec['code_bytes']=len(code.encode())
   replay=await api.post(base+'/token',form=form);assert replay.status==400;rec['code_replay_status']=replay.status
   refresh={'grant_type':'refresh_token','client_id':client_id,'refresh_token':tokens['refresh_token'],'resource':base+'/mcp'}
   response=await api.post(base+'/token',form=refresh);assert response.status==200,(response.status,await response.text())
   rotated=await response.json();assert rotated['refresh_token']!=tokens['refresh_token'];rec['refresh_status']=response.status
   response=await api.post(base+'/token',form=refresh);assert response.status==400;rec['refresh_replay_status']=response.status
   headers={'Authorization':'Bearer '+rotated['access_token'],'Accept':'application/json, text/event-stream'}
   for method,parameters in [('initialize',{'protocolVersion':'2025-06-18','capabilities':{},'clientInfo':{'name':'lit7449-verification','version':'1'}}),('tools/list',{})]:
    response=await api.post(base+'/mcp',headers=headers,data={'jsonrpc':'2.0','id':1,'method':method,'params':parameters})
    body=rpc(await response.text());assert response.status==200 and 'result' in body,(response.status,body)
    rec[method]={'status':response.status,'result':body['result']}
   tool=next(t['name'] for t in rec['tools/list']['result']['tools'] if t['name'].endswith('microsoft_docs_search'))
   response=await api.post(base+'/mcp',headers=headers,data={'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':tool,'arguments':{'query':'Azure resource groups'}}})
   body=rpc(await response.text());assert response.status==200 and 'result' in body and not body['result'].get('isError'),(response.status,body)
   rec['tools/call']={'status':response.status,'result':body['result']}
   await page.close()
   print(case,'PASS: login, consent, PKCE, refresh, replay rejection, public MCP list/call',flush=True)
   (out/'live-results.json').write_text(json.dumps(results,indent=2))
  (out/'live-results.json').write_text(json.dumps(results,indent=2))
  await browser.close()
asyncio.run(main())
