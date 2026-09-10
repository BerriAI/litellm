import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright,expect
from seed import request,root
from urllib.error import HTTPError

async def check_ui():
    async with async_playwright() as pw:
        browser=await pw.chromium.connect_over_cdp('http://127.0.0.1:9713')
        page=browser.contexts[0].pages[0]
        await page.goto('http://localhost:47135/ui/mcp-servers/')
        await page.get_by_title('preview_url',exact=True).click()
        await page.get_by_role('tab',name='Settings',exact=True).click()
        await expect(page.get_by_text('Unable to load tools',exact=True)).to_be_visible(timeout=30000)
        calls=[]
        def record(req):
            if '/mcp-rest/test/tools/list' in req.url: calls.append(req.method)
        page.on('request',record)
        await page.get_by_label('MCP Server URL',exact=True).fill('http://litellm-7135-upstream-1:8080/mcp')
        message=page.get_by_text('The server origin changed. Enter credentials and replace or remove saved static headers to preview tools.',exact=True)
        await expect(message).to_be_visible()
        await page.wait_for_timeout(1000)
        assert calls==[]
        await page.get_by_label('Authentication Value',exact=True).fill('preview:correct')
        await expect(page.get_by_role('button',name='Flat List',exact=True)).to_be_visible(timeout=15000)
        await page.get_by_role('button',name='Flat List',exact=True).click()
        await expect(page.get_by_text('echo',exact=True)).to_be_visible()
        page.remove_listener('request',record)
        await browser.close()
        return {'guardVisible':True,'requestsBeforeCredential':0,'requestsAfterCredential':len(calls),'toolVisible':'echo'}

sid=json.loads((root/'servers.json').read_text())['url']
payload={'server_id':sid,'url':'http://litellm-7135-upstream-1:8080/mcp','transport':'http','auth_type':'basic'}
try:
    status,result=request('POST','/mcp-rest/test/tools/list',payload)
except HTTPError as error:
    status,result=error.code,json.load(error)
assert status != 200 or result.get('error') or not result.get('tools')
explicit_status,explicit=request('POST','/mcp-rest/test/tools/list',{**payload,'credentials':{'auth_value':'preview:correct'}})
assert explicit_status==200 and explicit['tools'][0]['name']=='echo'
evidence={'commit':(root/'after-sha').read_text().strip(),'withoutExplicitCredential':{'status':status,'toolCount':len(result.get('tools',[]))},'withExplicitCredential':{'status':explicit_status,'tools':[t['name'] for t in explicit['tools']]},'ui':asyncio.run(check_ui())}
(root/'origin-boundary.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence))
