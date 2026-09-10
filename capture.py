import asyncio
import json
import sys
from pathlib import Path
from playwright.async_api import async_playwright, expect

root = Path(__file__).resolve().parent
revision = sys.argv[1]

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp('http://127.0.0.1:9713')
        page = browser.contexts[0].pages[0]
        await page.set_viewport_size({'width': 1440, 'height': 1300})
        for case in ['auth', 'url', 'headers']:
            await page.goto('http://localhost:47135/ui/mcp-servers/')
            await page.get_by_title('preview_' + case, exact=True).click()
            await page.get_by_role('tab', name='Settings', exact=True).click()
            await expect(page.get_by_text('Unable to load tools', exact=True)).to_be_visible(timeout=30000)
            await page.wait_for_timeout(1500)
            records = []
            issued = set()
            def note_request(request):
                issued.add(request)
            async def record(response):
                if '/mcp-rest/' not in response.url:
                    return
                req = response.request
                if req not in issued:
                    return
                if '/tools/list' not in response.url:
                    return
                raw = req.post_data_json if req.method == 'POST' else None
                entry = {'method': req.method, 'url': response.url, 'status': response.status}
                if raw:
                    entry['config'] = {k: raw.get(k) for k in ['url', 'auth_type', 'server_id']}
                    entry['credentialsProvided'] = bool(raw.get('credentials'))
                    entry['staticHeaderNames'] = list(raw.get('static_headers', {}))
                try:
                    body = await response.json()
                    entry['tools'] = [t['name'] for t in body.get('tools', [])]
                    entry['error'] = body.get('error')
                    entry['message'] = body.get('message')
                except Exception:
                    entry['bodyUnavailable'] = True
                records.append(entry)
            page.on('request', note_request)
            page.on('response', record)
            if case == 'auth':
                await page.get_by_label('Authentication', exact=True).click()
                await page.get_by_role('option', name='Basic Auth', exact=True).click()
                await page.get_by_label('Authentication Value', exact=True).fill('preview:correct')
            elif case == 'url':
                await page.get_by_label('MCP Server URL', exact=True).fill('http://upstream:8080/mcp')
            else:
                await page.get_by_role('button', name='Permission Management / Access Control').click()
                await page.get_by_placeholder('Header value', exact=True).fill('correct')
            if revision == 'after':
                await expect(page.get_by_role('button', name='Flat List', exact=True)).to_be_visible(timeout=15000)
                await page.get_by_role('button', name='Flat List', exact=True).click()
                await expect(page.get_by_text('echo', exact=True)).to_be_visible()
                await expect(page.get_by_text('Unable to load tools', exact=True)).not_to_be_visible()
            else:
                await page.wait_for_timeout(1800)
                await expect(page.get_by_text('Unable to load tools', exact=True)).to_be_visible()
            await page.get_by_label('MCP Server URL', exact=True).evaluate("el => el.scrollIntoView({block: 'start'})")
            if case == 'headers':
                await page.get_by_placeholder('Header value', exact=True).evaluate("el => el.scrollIntoView({block: 'center'})")
            await page.screenshot(path=str(root / f'{revision}-{case}.png'))
            page.remove_listener('request', note_request)
            page.remove_listener('response', record)
            result = {'commit': (root / f'{revision}-sha').read_text().strip(), 'case': case, 'requestsAfterEdit': records, 'visibleTools': await page.get_by_text('echo', exact=True).count(), 'errorVisible': await page.get_by_text('Unable to load tools', exact=True).is_visible()}
            (root / f'{revision}-{case}-network.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
        await browser.close()

asyncio.run(main())
