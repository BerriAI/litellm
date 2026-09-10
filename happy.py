import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright, expect
from seed import request

root = Path(__file__).resolve().parent

async def save_corrected_auth():
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp('http://127.0.0.1:9713')
        page = browser.contexts[0].pages[0]
        await page.goto('http://localhost:47135/ui/mcp-servers/')
        await page.get_by_title('preview_auth', exact=True).click()
        await page.get_by_role('tab', name='Settings', exact=True).click()
        await page.get_by_label('Authentication', exact=True).click()
        await page.get_by_role('option', name='Basic Auth', exact=True).click()
        await page.get_by_label('Authentication Value', exact=True).fill('preview:correct')
        await expect(page.get_by_role('button', name='Flat List', exact=True)).to_be_visible(timeout=15000)
        async with page.expect_response(lambda response: '/v1/mcp/server' in response.url and response.request.method == 'PUT') as response:
            await page.get_by_role('button', name='Save Changes', exact=True).click()
        status = (await response.value).status
        await browser.close()
        return status

if __name__ == '__main__':
    saved_status = asyncio.run(save_corrected_auth())
    server_id = json.loads((root / 'servers.json').read_text())['auth']
    list_status, tools = request('GET', '/mcp-rest/tools/list?server_id=' + server_id)
    tool_name = tools['tools'][0]['name']
    payload = {'server_id': server_id, 'name': tool_name, 'arguments': {'message': 'LIT-7135 tool call verified'}}
    call_status, result = request('POST', '/mcp-rest/tools/call', payload)
    evidence = {'commit': (root/'after-sha').read_text().strip(), 'saveStatus': saved_status, 'listStatus': list_status, 'tools': [t['name'] for t in tools['tools']], 'callRequest': payload, 'callStatus': call_status, 'callResult': result}
    (root/'happy-path.json').write_text(json.dumps(evidence, indent=2))
    assert saved_status == 202 and list_status == 200 and call_status == 200
    assert not result.get('isError', False)
    assert 'LIT-7135 tool call verified' in json.dumps(result)
    print(json.dumps(evidence))
