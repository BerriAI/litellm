import asyncio,sys
from pathlib import Path
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as pw:
  b=await pw.chromium.connect_over_cdp('http://127.0.0.1:9748')
  p=next(p for p in b.contexts[0].pages if '127.0.0.1:4000' in p.url)
  if sys.argv[1]=='login':
   settings=dict(line.split('=',1) for line in Path('.env').read_text().splitlines() if '=' in line)
   await p.get_by_placeholder('Enter your username').fill('admin')
   await p.get_by_placeholder('Enter your password').fill(settings['LITELLM_MASTER_KEY'])
   await p.get_by_role('button',name='Login',exact=True).click()
  elif sys.argv[1]=='approve':
   await p.get_by_role('button',name='Finish connecting',exact=True).click()
  await p.wait_for_timeout(1200)
  print((await p.locator('body').inner_text())[:5000])
  if sys.argv[1]!='approve':await p.screenshot(path='after/vscode-desktop-consent.png',full_page=True)
  await b.close()
asyncio.run(main())
