import { expect, test, type Page } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  ADMIN_VIEWER_STORAGE_PATH,
  INTERNAL_USER_STORAGE_PATH,
  INTERNAL_VIEWER_STORAGE_PATH,
  TEAM_ADMIN_STORAGE_PATH,
} from "../../constants";
import { Role, users } from "../../fixtures/users";
import { rootPath } from "../../helpers/traffic";

const ROUTE = "cost-optimization";
const TAB_PARAM = "tab";
const ROLE_PATH = "/management/v1/users/me";
const OVERALL_REQUEST = "/user/daily/activity/aggregated";
const GATED_TAB = {
  slug: "caching",
  name: "Prompt Caching",
  request: "/cost_optimization/prompt_caching/requests",
};
const DEFAULT_TAB = "Overall";

const ROLES = [
  { role: "proxy_admin", storageState: ADMIN_STORAGE_PATH, seesGated: true },
  {
    role: "proxy_admin_viewer",
    storageState: ADMIN_VIEWER_STORAGE_PATH,
    seesGated: true,
  },
  {
    role: "internal_user",
    storageState: INTERNAL_USER_STORAGE_PATH,
    seesGated: false,
  },
  {
    role: "internal_user_viewer",
    storageState: INTERNAL_VIEWER_STORAGE_PATH,
    seesGated: false,
  },
  {
    role: "team_admin",
    storageState: TEAM_ADMIN_STORAGE_PATH,
    seesGated: false,
  },
] as const;

const pageUrl = (params: Record<string, string>): string => {
  const query = new URLSearchParams(params).toString();
  return `${rootPath()}/ui/${ROUTE}/${query ? `?${query}` : ""}`;
};

const onRoute = (url: URL): boolean =>
  url.pathname.replace(/\/+$/, "").endsWith(`/ui/${ROUTE}`);

const pathEndsWith =
  (suffix: string) =>
  (url: URL): boolean =>
    url.pathname.endsWith(suffix);

const selectedTab = (page: Page, name: string) =>
  page.getByRole("tab", { name, exact: true, selected: true });

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
      window.history[method] = (
        data: unknown,
        unused: string,
        url?: string | URL | null,
      ) => {
        original(data, unused, url);
        writes.push(window.location.href);
      };
    });
  });

const urlWrites = (page: Page): Promise<string[]> =>
  page.evaluate(() => [
    ...(window as unknown as { __urlWrites: string[] }).__urlWrites,
  ]);

const paramIn = (href: string, key: string): string | null =>
  new URL(href).searchParams.get(key);

const requestedPaths = (page: Page): string[] => {
  const seen: string[] = [];
  page.on("request", (request) => seen.push(new URL(request.url()).pathname));
  return seen;
};

const logIn = async (
  page: Page,
  credentials: { email: string; password: string },
): Promise<void> => {
  await page.getByPlaceholder("Enter your username").fill(credentials.email);
  await page.getByPlaceholder("Enter your password").fill(credentials.password);
  await page.getByRole("button", { name: "Login", exact: true }).click();
};

test.describe(`${ROUTE} tab links by role`, () => {
  for (const { role, storageState, seesGated } of ROLES) {
    test.describe(role, () => {
      test.use({ storageState });

      test(`?${TAB_PARAM}=${GATED_TAB.slug} opens ${seesGated ? "that tab" : "the default tab and keeps the param"}`, async ({
        page,
      }) => {
        const requests = requestedPaths(page);
        const overallLoaded = page.waitForResponse((response) =>
          pathEndsWith(OVERALL_REQUEST)(new URL(response.url())),
        );
        await page.goto(pageUrl({ [TAB_PARAM]: GATED_TAB.slug }));
        await overallLoaded;

        await expect(
          selectedTab(page, seesGated ? GATED_TAB.name : DEFAULT_TAB),
        ).toBeVisible();
        await expect(page).toHaveURL(
          (url) =>
            onRoute(url) && url.searchParams.get(TAB_PARAM) === GATED_TAB.slug,
        );
        if (seesGated) {
          await expect
            .poll(() =>
              requests.some((path) => path.endsWith(GATED_TAB.request)),
            )
            .toBe(true);
          return;
        }

        await expect(
          page.getByRole("tab", { name: GATED_TAB.name, exact: true }),
        ).toHaveCount(0);
        expect(requests).toContain(OVERALL_REQUEST);
        expect(
          requests.filter((path) => path.endsWith(GATED_TAB.request)),
        ).toEqual([]);
      });
    });
  }
});

test.describe(`${ROUTE} deep link while the role loads`, () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("never drops the param while the role is pending and opens the tab once it lands", async ({
    page,
  }) => {
    await recordUrlWrites(page);
    const role = await holdRoute(page, ROLE_PATH);
    await page.goto(pageUrl({ [TAB_PARAM]: GATED_TAB.slug }));
    await role.arrived;

    await expect(page).toHaveURL(
      (url) => url.searchParams.get(TAB_PARAM) === GATED_TAB.slug,
    );
    role.release();
    await expect(selectedTab(page, GATED_TAB.name)).toBeVisible();
    await expect(page).toHaveURL(
      (url) => url.searchParams.get(TAB_PARAM) === GATED_TAB.slug,
    );
    expect(
      (await urlWrites(page)).filter(
        (href) => paramIn(href, TAB_PARAM) !== GATED_TAB.slug,
      ),
    ).toEqual([]);

    await page.getByRole("tab", { name: DEFAULT_TAB, exact: true }).click();
    await expect(page).toHaveURL(
      (url) => onRoute(url) && !url.searchParams.has(TAB_PARAM),
    );
    await expect
      .poll(async () =>
        (await urlWrites(page)).some(
          (href) => paramIn(href, TAB_PARAM) === null,
        ),
      )
      .toBe(true);
  });
});

test.describe(`${ROUTE} deep link through login`, () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("a logged-out visitor lands on the linked tab after logging in", async ({
    page,
  }) => {
    await page.goto(pageUrl({ [TAB_PARAM]: GATED_TAB.slug }));
    await expect(page).toHaveURL(/\/ui\/login/);

    await logIn(page, users[Role.ProxyAdmin]);
    await expect(selectedTab(page, GATED_TAB.name)).toBeVisible();
    await expect(page).toHaveURL(
      (url) =>
        onRoute(url) && url.searchParams.get(TAB_PARAM) === GATED_TAB.slug,
    );
  });
});

test.describe(`${ROUTE} failed role request`, () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("sends the user to login and back to the linked tab", async ({
    page,
  }) => {
    const failingRole = pathEndsWith(ROLE_PATH);
    await page.route(failingRole, (route) =>
      route.fulfill({ status: 500, json: { detail: "e2e role failure" } }),
    );
    await page.goto(pageUrl({ [TAB_PARAM]: GATED_TAB.slug }));
    await expect(page).toHaveURL(/\/ui\/login/);

    await page.unroute(failingRole);
    await logIn(page, users[Role.ProxyAdmin]);
    await expect(selectedTab(page, GATED_TAB.name)).toBeVisible();
    await expect(page).toHaveURL(
      (url) =>
        onRoute(url) && url.searchParams.get(TAB_PARAM) === GATED_TAB.slug,
    );
  });
});
