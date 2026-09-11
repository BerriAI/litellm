import asyncio,sys
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).parent
async def main():
 env=dict(line.split('=',1) for line in (ROOT/'local.env').read_text().splitlines())
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
  for label,port in [('before',47498),('after',47499)]:
   context=await browser.new_context(viewport={'width':1440,'height':1100})
   page=await context.new_page()
   await page.goto(f'http://127.0.0.1:{port}/ui/',wait_until='networkidle')
   await page.get_by_role('textbox',name='Username').fill('admin')
   await page.locator('input[type=password]').fill(env['LITELLM_MASTER_KEY'])
   await page.get_by_role('button',name='Login',exact=True).click()
   await page.wait_for_timeout(1500)
   await page.goto(f'http://127.0.0.1:{port}/ui/?page=mcp-servers',wait_until='networkidle')
   await page.get_by_text('figma',exact=True).first.click()
   await page.get_by_role('tab',name='Settings',exact=True).click()
   await page.get_by_role('button',name='Authorize & Fetch Tools (browser-only)',exact=True).click()
   expected='upstream registration failed with HTTP 403' if label=='before' else 'the upstream authorization server refused dynamic client registration'
   await page.get_by_text(expected,exact=False).first.wait_for(timeout=20000)
   section=page.locator('div.border-dashed').filter(has=page.get_by_role('button',name='Authorize & Fetch Tools (browser-only)',exact=True))
   await section.screenshot(path=str(ROOT/f'{label}-oauth.png'))
   print(label, (await section.inner_text())[-900:])
   await context.close()
  await browser.close()
asyncio.run(main())
