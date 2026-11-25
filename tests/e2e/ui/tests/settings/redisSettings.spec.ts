import { expect, test } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { rootPath } from "../../helpers/traffic";

const cases = [
  {
    tab: "Cache Settings",
    path: "/cache/settings",
    prefill: { current_values: { port: 6380 } },
    body: {
      cache_settings: {
        type: "redis",
        host: "127.0.0.1",
        port: "1",
        ssl: false,
        ssl_check_hostname: false,
      },
    },
  },
  {
    tab: "Coordination Redis",
    path: "/coordination_redis/settings",
    prefill: { values: { port: 6380 }, fields: [], source: null },
    body: { settings: { host: "127.0.0.1", port: 1, ssl: false } },
  },
] as const;

test.describe("Redis settings", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  for (const { tab, path, prefill, body } of cases) {
    test(`${tab} blocks an invalid port and tests the corrected settings`, async ({
      page,
    }) => {
      const savePath = `${rootPath()}${path}`;
      const testPath = `${savePath}/test`;
      await page.route(
        (url) => url.pathname === savePath,
        (route) =>
          route.request().method() === "GET"
            ? route.fulfill({ json: prefill })
            : route.continue(),
      );
      const writes: string[] = [];
      page.on("request", (request) => {
        const pathname = new URL(request.url()).pathname;
        if (
          request.method() === "POST" &&
          (pathname === savePath || pathname === testPath)
        ) {
          writes.push(pathname);
        }
      });

      await page.goto(`${rootPath()}/ui/caching/`);
      await page.getByRole("tab", { name: tab, exact: true }).click();
      const panel = page.getByRole("tabpanel", { name: tab, exact: true });
      const port = panel.getByRole("textbox", { name: "Port", exact: true });
      await expect(port).toHaveValue("6380");
      await panel.getByRole("combobox").first().click();
      await page
        .getByRole("option", { name: "Node (Single Instance)", exact: true })
        .click();
      await panel
        .getByRole("textbox", { name: "Redis URL", exact: true })
        .fill("");
      await panel
        .getByRole("textbox", { name: "Host", exact: true })
        .fill("127.0.0.1");
      const testConnection = panel.getByRole("button", {
        name: "Test Connection",
        exact: true,
      });
      const invalidPort = panel
        .getByRole("alert")
        .filter({ hasText: "Port must be an integer between 1 and 65535" });

      await port.fill("65536");
      await testConnection.click();
      await expect(invalidPort).toBeVisible();
      await panel
        .getByRole("button", { name: "Save Changes", exact: true })
        .click();

      await port.fill("1");
      const response = page.waitForResponse(
        (result) =>
          result.request().method() === "POST" &&
          new URL(result.url()).pathname === testPath,
      );
      await testConnection.click();
      expect((await response).request().postDataJSON()).toEqual(body);
      await expect(invalidPort).toHaveCount(0);
      await expect(page.getByText(/^Connection test failed:/)).toBeVisible();
      expect(writes).toEqual([testPath]);
    });
  }
});
