import asyncio,json,sys
from pathlib import Path
from playwright.async_api import async_playwright
root=Path(__file__).parent
async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.connect_over_cdp('http://127.0.0.1:9489')
  page=browser.contexts[0].pages[0]
  await page.set_viewport_size({'width':1440,'height':1400})
  await page.goto('http://localhost:44896/ui/playground/')
  await page.get_by_placeholder('Select an endpoint',exact=True).click()
  await page.get_by_role('option',name='/mcp-rest/tools/call',exact=True).click()
  await page.get_by_placeholder('Select MCP server',exact=True).click()
  await page.get_by_role('option',name='remote_none',exact=True).click()
  await page.get_by_placeholder('Select a tool to call',exact=True).click()
  await page.get_by_role('option',name='getauthenticateduser',exact=True).click()
  await page.get_by_role('button',name='Clear Chat',exact=True).click()
  async with page.expect_response(lambda r:'/mcp-rest/tools/call' in r.url and r.request.method=='POST',timeout=60000) as result:
   await page.get_by_role('button',name='Send message',exact=True).click()
  response=await result.value;body=await response.json()
  (root/(sys.argv[1]+'-tool.json')).write_text(json.dumps({'status':response.status,'request':response.request.post_data_json,'isError':body.get('isError'),'content_returned':bool(body.get('content'))},indent=2))
  await page.wait_for_timeout(1000)
  await page.screenshot(path=str(root/(sys.argv[1]+'-tool.png')))
  print('Saved live Playground tool screenshot')
  await browser.close()
asyncio.run(main())
