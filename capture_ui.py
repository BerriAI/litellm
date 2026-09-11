import asyncio,json,sys
from pathlib import Path
from playwright.async_api import async_playwright
root=Path(__file__).parent
async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.connect_over_cdp('http://127.0.0.1:9489')
  page=browser.contexts[0].pages[0]
  await page.set_viewport_size({'width':1440,'height':1600})
  await page.goto('http://localhost:44896/ui/mcp-servers/')
  await page.wait_for_timeout(15000)
  card=page.locator('div[role="button"]').filter(has=page.get_by_title('remote_none',exact=True))
  await card.get_by_role('button',name='Server actions',exact=True).click()
  async with page.expect_response(lambda r:'/v1/mcp/server/health?' in r.url,timeout=40000) as result:
   await page.get_by_role('menuitem',name='Test Connection',exact=True).click()
  response=await result.value
  (root/(sys.argv[1]+'-test-connection.json')).write_text(json.dumps({'url':response.url,'status':response.status,'body':await response.json()},indent=2))
  await page.mouse.move(280,150)
  await page.wait_for_timeout(1500)
  await page.screenshot(path=str(root/(sys.argv[1]+'-health.png')))
  (root/(sys.argv[1]+'-ui.txt')).write_text(await page.locator('body').inner_text())
  for name in ['byok_default','remote_missing','remote_invalid','remote_timeout','remote_large','remote_compressed']:
   target=page.locator('div[role="button"]').filter(has=page.get_by_title(name,exact=True))
   badge=target.get_by_text(__import__('re').compile(r'^(Healthy|Unhealthy|Unknown)$'))
   async with page.expect_response(lambda r:'/v1/mcp/server/health?' in r.url,timeout=40000):
    await badge.click()
   await page.wait_for_timeout(500)
   await badge.hover()
   await page.wait_for_timeout(800)
   await page.screenshot(path=str(root/(sys.argv[1]+'-'+name+'.png')))
   await page.mouse.move(280,150)
  print('Saved dashboard overview and four scenario screenshots')
  await browser.close()
asyncio.run(main())
