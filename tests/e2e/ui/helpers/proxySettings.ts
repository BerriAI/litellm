import { expect, type APIRequestContext, type APIResponse, type Page } from "@playwright/test";
import { masterKey, rootPath } from "./traffic";

export async function writeProxySettings(request: APIRequestContext, name: string, settings: object) {
  const response = await request.patch(`${rootPath()}/update/${name}`, {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: settings,
  });
  expect(response.ok(), await response.text()).toBe(true);
}

export async function seedProxySettings(request: APIRequestContext, page: Page, name: string, settings: object) {
  await writeProxySettings(request, name, settings);
  await page.route(
    `**/get/${name}`,
    async (route) => {
      const headers = { ...(await route.request().allHeaders()), Connection: "close" };
      let response: APIResponse | undefined;
      try {
        await expect
          .poll(
            async () => {
              await response?.dispose();
              response = await route.fetch({ headers });
              return response.ok() ? (await response.json()).values : response.status();
            },
            { timeout: 30000, message: `Seed ${name}` },
          )
          .toEqual(settings);
        await route.fulfill({ response });
      } catch (error) {
        await route.abort();
        throw error;
      } finally {
        await response?.dispose();
      }
    },
    { times: 1 },
  );
}
