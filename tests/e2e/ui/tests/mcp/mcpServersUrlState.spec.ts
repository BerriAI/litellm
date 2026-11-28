import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  ADMIN_VIEWER_STORAGE_PATH,
  INTERNAL_USER_STORAGE_PATH,
  INTERNAL_VIEWER_STORAGE_PATH,
  TEAM_ADMIN_STORAGE_PATH,
} from "../../constants";
import { Role, users } from "../../fixtures/users";
import { runWithCleanup } from "../../helpers/roundTrip";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";

const ROLE_PATH = "/management/v1/users/me";
const SERVER_LIST_PATH = "/v1/mcp/server";
const TOOL_LIST_PATH = "/mcp-rest/tools/list";
const DEFAULT_TAB = "All Servers";
const DEFAULT_SERVER_TAB = "Overview";

const GATED_PAGE_TABS = [
  { slug: "semantic-filter", name: "Semantic Filter", request: "/get/mcp_semantic_filter_settings" },
  { slug: "tool-search", name: "Tool Search", request: "/get/mcp_tool_search_settings" },
  { slug: "network-settings", name: "Network Settings", request: "/v1/mcp/network/client-ip" },
  { slug: "submitted", name: "Submitted MCPs", request: "/v1/mcp/server/submissions" },
  { slug: "connections", name: "Live Connections", request: "/v1/mcp/sessions" },
] as const;

const GATED_SERVER_TABS = [
  { slug: "settings", name: "Settings" },
  { slug: "user-credentials", name: "User Credentials" },
] as const;

const ROLES = [
  { role: "proxy_admin", storageState: ADMIN_STORAGE_PATH, seesGated: true },
  { role: "proxy_admin_viewer", storageState: ADMIN_VIEWER_STORAGE_PATH, seesGated: true },
  { role: "internal_user", storageState: INTERNAL_USER_STORAGE_PATH, seesGated: false },
  { role: "internal_user_viewer", storageState: INTERNAL_VIEWER_STORAGE_PATH, seesGated: false },
  { role: "team_admin", storageState: TEAM_ADMIN_STORAGE_PATH, seesGated: false },
] as const;

const FAKE_TOOL = "lookup_docs";

const adminHeaders = (): Record<string, string> => ({ Authorization: `Bearer ${masterKey()}` });

const pageUrl = (params: Record<string, string>): string => {
  const query = new URLSearchParams(params).toString();
  return `${rootPath()}/ui/mcp-servers/${query ? `?${query}` : ""}`;
};

const onRoute = (url: URL): boolean => url.pathname.replace(/\/+$/, "").endsWith("/mcp-servers");

const pathEndsWith =
  (suffix: string) =>
  (url: URL): boolean =>
    url.pathname.endsWith(suffix);

const userCredentialsPath = (serverId: string): string => `/v1/mcp/server/${serverId}/user-credentials`;

const selectedTab = (page: Page, name: string) => page.getByRole("tab", { name, exact: true, selected: true });

const backToServers = (page: Page) => page.getByRole("button", { name: /Back to All Servers/i });

interface Deferred {
  promise: Promise<void>;
  resolve: () => void;
}

const deferred = (): Deferred => {
  const box: { resolve: () => void } = { resolve: () => undefined };
  const promise = new Promise<void>((resolve) => {
    box.resolve = resolve;
  });
  return { promise, resolve: () => box.resolve() };
};

interface HeldRoute {
  arrived: Promise<void>;
  release: () => void;
}

const holdRoute = async (page: Page, suffix: string): Promise<HeldRoute> => {
  const arrival = deferred();
  const gate = deferred();
  await page.route(pathEndsWith(suffix), async (route) => {
    arrival.resolve();
    await gate.promise;
    await route.fallback();
  });
  return { arrived: arrival.promise, release: gate.resolve };
};

const recordUrlWrites = (page: Page): Promise<void> =>
  page.addInitScript(() => {
    const writes: string[] = [];
    Object.defineProperty(window, "__urlWrites", { value: writes });
    (["pushState", "replaceState"] as const).forEach((method) => {
      const original = window.history[method].bind(window.history);
      window.history[method] = (data: unknown, unused: string, url?: string | URL | null) => {
        original(data, unused, url);
        writes.push(window.location.href);
      };
    });
  });

const urlWrites = (page: Page): Promise<string[]> =>
  page.evaluate(() => [...(window as unknown as { __urlWrites: string[] }).__urlWrites]);

const paramIn = (href: string, key: string): string | null => new URL(href).searchParams.get(key);

const requested = (paths: readonly string[], suffix: string): boolean => paths.some((path) => path.endsWith(suffix));

const requestedPaths = (page: Page): string[] => {
  const seen: string[] = [];
  page.on("request", (request) => seen.push(new URL(request.url()).pathname));
  return seen;
};

const openPage = async (page: Page, params: Record<string, string>): Promise<void> => {
  const roleLoaded = page.waitForResponse((response) => pathEndsWith(ROLE_PATH)(new URL(response.url())));
  const serversLoaded = page.waitForResponse((response) => pathEndsWith(SERVER_LIST_PATH)(new URL(response.url())));
  await page.goto(pageUrl(params));
  await Promise.all([roleLoaded, serversLoaded]);
};

const logIn = async (page: Page, credentials: { email: string; password: string }): Promise<void> => {
  await page.getByPlaceholder("Enter your username").fill(credentials.email);
  await page.getByPlaceholder("Enter your password").fill(credentials.password);
  await page.getByRole("button", { name: "Login", exact: true }).click();
};

const createServer = async (request: APIRequestContext, name: string): Promise<string> => {
  const response = await request.post(`${rootPath()}${SERVER_LIST_PATH}`, {
    headers: adminHeaders(),
    data: {
      server_name: name,
      alias: name,
      url: "http://127.0.0.1:9/mcp",
      transport: "http",
      auth_type: "none",
      allow_all_keys: true,
    },
  });
  expect(response.ok(), `POST ${SERVER_LIST_PATH} failed (${response.status()}): ${await response.text()}`).toBe(true);
  const { server_id: id } = (await response.json()) as { server_id?: string };
  expect(id, `POST ${SERVER_LIST_PATH} returned no server_id`).toBeTruthy();
  return id ?? "";
};

const deleteServer = async (request: APIRequestContext, id: string): Promise<boolean> =>
  (await request.delete(`${rootPath()}${SERVER_LIST_PATH}/${id}`, { headers: adminHeaders() })).ok();

const withServer = async (request: APIRequestContext, run: (id: string, name: string) => Promise<void>) => {
  const name = `e2e_url_state_${uniqueSuffix().replace(/-/g, "_")}`;
  const id = await createServer(request, name);
  await runWithCleanup(
    () => run(id, name),
    () => deleteServer(request, id),
  );
};

const fakeToolList = (page: Page, serverName: string): Promise<void> =>
  page.route(pathEndsWith(TOOL_LIST_PATH), (route) =>
    route.fulfill({
      json: {
        tools: [
          {
            name: FAKE_TOOL,
            description: "Looks up a docs page",
            inputSchema: { type: "object", properties: { query: { type: "string" } } },
            mcp_info: { server_name: serverName },
          },
        ],
      },
    }),
  );

test.describe("mcp-servers tab links by role", () => {
  for (const { role, storageState, seesGated } of ROLES) {
    test.describe(role, () => {
      test.use({ storageState });

      test(`gated page and server tab links ${seesGated ? "open their tab" : "fall back and keep the param"}`, async ({
        page,
        request,
      }) => {
        await withServer(request, async (serverId) => {
          const requests = requestedPaths(page);

          for (const tab of GATED_PAGE_TABS) {
            await openPage(page, { tab: tab.slug });
            await expect(selectedTab(page, seesGated ? tab.name : DEFAULT_TAB)).toBeVisible();
            await expect(page).toHaveURL((url) => onRoute(url) && url.searchParams.get("tab") === tab.slug);
            if (!seesGated) await expect(page.getByRole("tab", { name: tab.name, exact: true })).toHaveCount(0);
          }

          for (const tab of GATED_SERVER_TABS) {
            await openPage(page, { server: serverId, server_tab: tab.slug });
            await expect(backToServers(page)).toBeVisible();
            await expect(selectedTab(page, seesGated ? tab.name : DEFAULT_SERVER_TAB)).toBeVisible();
            await expect(page).toHaveURL(
              (url) => url.searchParams.get("server") === serverId && url.searchParams.get("server_tab") === tab.slug,
            );
            if (!seesGated) await expect(page.getByRole("tab", { name: tab.name, exact: true })).toHaveCount(0);
          }

          if (seesGated) return;
          expect(requested(requests, ROLE_PATH)).toBe(true);
          expect(requested(requests, SERVER_LIST_PATH)).toBe(true);
          const adminOnly = [...GATED_PAGE_TABS.map((tab) => tab.request), userCredentialsPath(serverId)];
          expect(requests.filter((path) => adminOnly.some((gated) => path.endsWith(gated)))).toEqual([]);
        });
      });
    });
  }
});

test.describe("mcp-servers deep links while the role loads", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("a page tab link keeps its param while the role is pending and opens once it lands", async ({ page }) => {
    await recordUrlWrites(page);
    const role = await holdRoute(page, ROLE_PATH);
    await page.goto(pageUrl({ tab: "semantic-filter" }));
    await role.arrived;

    await expect(page).toHaveURL((url) => url.searchParams.get("tab") === "semantic-filter");
    role.release();
    await expect(selectedTab(page, "Semantic Filter")).toBeVisible();
    await expect(page).toHaveURL((url) => url.searchParams.get("tab") === "semantic-filter");
    expect((await urlWrites(page)).filter((href) => paramIn(href, "tab") !== "semantic-filter")).toEqual([]);

    await page.getByRole("tab", { name: DEFAULT_TAB, exact: true }).click();
    await expect(page).toHaveURL((url) => onRoute(url) && !url.searchParams.has("tab"));
  });

  test("a server tab link keeps its param while the role is pending and opens once it lands", async ({
    page,
    request,
  }) => {
    await withServer(request, async (serverId) => {
      await recordUrlWrites(page);
      const role = await holdRoute(page, ROLE_PATH);
      await page.goto(pageUrl({ server: serverId, server_tab: "settings" }));
      await role.arrived;

      await expect(page).toHaveURL((url) => url.searchParams.get("server_tab") === "settings");
      role.release();
      await expect(selectedTab(page, "Settings")).toBeVisible();
      await expect(page).toHaveURL(
        (url) => url.searchParams.get("server") === serverId && url.searchParams.get("server_tab") === "settings",
      );
      const writes = await urlWrites(page);
      expect(writes.filter((href) => paramIn(href, "server_tab") !== "settings")).toEqual([]);
    });
  });

  test("an internal user on a held role ends on All Servers with the param kept", async ({ browser }) => {
    const context = await browser.newContext({ storageState: INTERNAL_USER_STORAGE_PATH });
    const page = await context.newPage();
    const requests = requestedPaths(page);
    const role = await holdRoute(page, ROLE_PATH);
    const serversLoaded = page.waitForResponse((response) => pathEndsWith(SERVER_LIST_PATH)(new URL(response.url())));
    await page.goto(pageUrl({ tab: "semantic-filter" }));
    await role.arrived;
    role.release();
    await serversLoaded;

    await expect(selectedTab(page, DEFAULT_TAB)).toBeVisible();
    await expect(page).toHaveURL((url) => url.searchParams.get("tab") === "semantic-filter");
    expect(requested(requests, SERVER_LIST_PATH)).toBe(true);
    expect(requests.filter((path) => path.endsWith("/get/mcp_semantic_filter_settings"))).toEqual([]);
    await context.close();
  });
});

test.describe("mcp-servers deep link through login", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("a logged-out visitor lands on the linked tab after logging in", async ({ page }) => {
    await page.goto(pageUrl({ tab: "tool-search" }));
    await expect(page).toHaveURL(/\/login/);

    await logIn(page, users[Role.ProxyAdmin]);
    await expect(selectedTab(page, "Tool Search")).toBeVisible();
    await expect(page).toHaveURL((url) => onRoute(url) && url.searchParams.get("tab") === "tool-search");
  });
});

test.describe("mcp-servers failed role request", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("sends the user to login and back to the linked tab", async ({ page }) => {
    const failingRole = pathEndsWith(ROLE_PATH);
    await page.route(failingRole, (route) => route.fulfill({ status: 500, json: { detail: "e2e role failure" } }));
    await page.goto(pageUrl({ tab: "network-settings" }));
    await expect(page).toHaveURL(/\/login/);

    await page.unroute(failingRole);
    await logIn(page, users[Role.ProxyAdmin]);
    await expect(selectedTab(page, "Network Settings")).toBeVisible();
    await expect(page).toHaveURL((url) => onRoute(url) && url.searchParams.get("tab") === "network-settings");
  });
});

test.describe("mcp-servers detail view history", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Back closes the open server and Forward reopens it", async ({ page, request }) => {
    await withServer(request, async (serverId, name) => {
      await openPage(page, {});
      await page.getByTestId("mcp-servers-grid").getByTitle(name, { exact: true }).click();
      await expect(page).toHaveURL((url) => url.searchParams.get("server") === serverId);
      await expect(backToServers(page)).toBeVisible();

      await page.goBack();
      await expect(page).toHaveURL((url) => onRoute(url) && !url.searchParams.has("server"));
      await expect(selectedTab(page, DEFAULT_TAB)).toBeVisible();
      await expect(backToServers(page)).toHaveCount(0);

      await page.goForward();
      await expect(page).toHaveURL((url) => url.searchParams.get("server") === serverId);
      await expect(backToServers(page)).toBeVisible();
    });
  });

  test("Back closes the picked tool and keeps the server on its tools tab", async ({ page, request }) => {
    await withServer(request, async (serverId, name) => {
      await fakeToolList(page, name);
      await openPage(page, { server: serverId, server_tab: "tools" });
      const toolCard = page.locator("h4.font-mono").filter({ hasText: new RegExp(`^${FAKE_TOOL}$`) });
      await toolCard.click();
      await expect(page).toHaveURL((url) => url.searchParams.get("tool") === FAKE_TOOL);

      await page.reload();
      await expect(selectedTab(page, "MCP Tools")).toBeVisible();
      await expect(page).toHaveURL((url) => url.searchParams.get("tool") === FAKE_TOOL);

      await page.goBack();
      await expect(page).toHaveURL(
        (url) =>
          url.searchParams.get("server") === serverId &&
          url.searchParams.get("server_tab") === "tools" &&
          !url.searchParams.has("tool"),
      );
      await expect(selectedTab(page, "MCP Tools")).toBeVisible();
    });
  });
});
