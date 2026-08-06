import test, { expect } from "@playwright/test";
import type {
  APIRequestContext,
  Page as PlaywrightPage,
} from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  INTERNAL_VIEWER_STORAGE_PATH,
} from "../../constants";
import { users } from "../../fixtures/users";
import { Role } from "../../fixtures/roles";

const MASTER_KEY = users[Role.ProxyAdmin].password;
const HOLD_SECONDS = 25;

/** Fires a completion the mock upstream holds open, so the request is still running while we assert. */
function startHeldCompletion(
  request: APIRequestContext,
  endUser: string,
): Promise<unknown> {
  return request.post("/v1/chat/completions", {
    headers: { Authorization: `Bearer ${MASTER_KEY}` },
    data: {
      model: "fake-openai-gpt-4",
      user: endUser,
      messages: [
        {
          role: "user",
          content: `e2e-hold-${HOLD_SECONDS}s active requests page`,
        },
      ],
    },
    timeout: (HOLD_SECONDS + 30) * 1000,
  });
}

async function openPage(page: PlaywrightPage): Promise<void> {
  await page.goto("/ui/active-requests/");
  await expect(
    page.getByRole("heading", { name: "Active Requests" }),
  ).toBeVisible();
}

test.describe("proxy admin active requests", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("should show a running request with its caller, then drop it once it finishes", async ({
    page,
    request,
  }) => {
    const endUser = `e2e-active-${Date.now()}`;
    const inFlight = startHeldCompletion(request, endUser);

    await openPage(page);
    await expect(page.getByText(endUser)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("fake-openai-gpt-4").first()).toBeVisible();

    await inFlight;

    await expect(page.getByText(endUser)).toBeHidden({ timeout: 20_000 });
  });

  test("should narrow the page to one caller when filtering by end user", async ({
    page,
    request,
  }) => {
    const mine = `e2e-mine-${Date.now()}`;
    const other = `e2e-other-${Date.now()}`;
    const inFlight = Promise.all([
      startHeldCompletion(request, mine),
      startHeldCompletion(request, other),
    ]);

    await openPage(page);
    await expect(page.getByText(mine)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(other)).toBeVisible();

    await page.getByPlaceholder("End User ID").fill(mine);

    await expect(page.getByText(other)).toBeHidden({ timeout: 15_000 });
    await expect(page.getByText(mine)).toBeVisible();

    await inFlight;
  });

  test("should regroup the chart by end user without leaving the page", async ({
    page,
    request,
  }) => {
    const endUser = `e2e-group-${Date.now()}`;
    const inFlight = startHeldCompletion(request, endUser);

    await openPage(page);
    await expect(page.getByText("Running requests by model")).toBeVisible();

    await page.getByRole("tab", { name: "End User", exact: true }).click();

    await expect(page.getByText("Running requests by end user")).toBeVisible();
    await expect(page).toHaveURL(/\/ui\/active-requests\/?($|\?)/);

    await inFlight;
  });

  test("should open a request's details and cancel it", async ({
    page,
    request,
  }) => {
    const endUser = `e2e-cancel-${Date.now()}`;
    const inFlight = startHeldCompletion(request, endUser);

    await openPage(page);
    await page.getByText(endUser).click();

    const panel = page.getByRole("dialog");
    await expect(panel.getByText("Request ID")).toBeVisible();
    await expect(
      panel.getByRole("link", { name: "Open in Logs" }),
    ).toBeVisible();

    await panel.getByRole("button", { name: "Cancel request" }).click();

    await expect(page.getByText(endUser)).toBeHidden({ timeout: 20_000 });
    await inFlight.catch(() => undefined);
  });

  test("should pause the poll so the table stops moving", async ({
    page,
    request,
  }) => {
    const endUser = `e2e-pause-${Date.now()}`;
    const inFlight = startHeldCompletion(request, endUser);

    await openPage(page);
    const updated = page.getByText(/^Updated /);
    await expect(updated).toBeVisible({ timeout: 15_000 });

    await page.getByRole("switch").click();
    const frozen = await updated.textContent();
    await page.waitForTimeout(12_000);

    expect(await updated.textContent()).toBe(frozen);
    await inFlight;
  });

  test("should keep the filter in the url so the view can be shared", async ({
    page,
    request,
  }) => {
    const endUser = `e2e-url-${Date.now()}`;
    const inFlight = startHeldCompletion(request, endUser);

    await openPage(page);
    await expect(page.getByText(endUser)).toBeVisible({ timeout: 15_000 });

    await page.getByPlaceholder("End User ID").fill(endUser);

    await expect(page).toHaveURL(new RegExp(`end_user_id=${endUser}`), {
      timeout: 10_000,
    });
    await inFlight;
  });

  test("should keep polling and stay interactive while a request runs", async ({
    page,
    request,
  }) => {
    const endUser = `e2e-poll-${Date.now()}`;
    const inFlight = startHeldCompletion(request, endUser);

    await openPage(page);
    const updated = page.getByText(/^Updated /);
    await expect(updated).toBeVisible({ timeout: 15_000 });
    const firstStamp = await updated.textContent();

    await expect
      .poll(async () => updated.textContent(), { timeout: 20_000 })
      .not.toBe(firstStamp);

    await page.getByRole("tab", { name: "Pod", exact: true }).click();
    await expect(page.getByText("Running requests by pod")).toBeVisible({
      timeout: 5_000,
    });

    await inFlight;
  });
});

test.describe("internal viewer active requests", () => {
  test.use({ storageState: INTERNAL_VIEWER_STORAGE_PATH });

  test("should not offer the page to a non-admin", async ({ page }) => {
    await page.goto("/ui");

    await expect(
      page.getByRole("link", { name: "Active Requests" }),
    ).toHaveCount(0);
  });
});
