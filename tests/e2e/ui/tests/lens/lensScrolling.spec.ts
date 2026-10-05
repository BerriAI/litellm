import { expect, test, type Locator, type Page } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

const metrics = (element: Locator) =>
  element.evaluate((el) => ({
    height: el.clientHeight,
    scrollHeight: el.scrollHeight,
    top: el.getBoundingClientRect().top,
    bottom: el.getBoundingClientRect().bottom,
  }));

async function expectTableOwnsScroll(page: Page, name: string) {
  const panel = page.getByRole("tabpanel", { name, exact: true });
  const table = panel.locator('[data-slot="inspector-table"]');
  const panelBox = await metrics(panel);
  const tableBox = await metrics(table);
  expect(
    panelBox.scrollHeight,
    "the tab must not add a second vertical scrollbar",
  ).toBe(panelBox.height);
  expect(tableBox.height).toBeGreaterThan(0);
  expect(
    tableBox.bottom,
    "the table scrollbar must fit inside its tab",
  ).toBeLessThanOrEqual(panelBox.bottom);
  expect(tableBox.scrollHeight).toBeGreaterThan(tableBox.height);

  const search = panel.getByRole("combobox");
  const searchTop = (await metrics(search)).top;
  await table.hover();
  await page.mouse.wheel(0, 500);
  await expect
    .poll(() => table.evaluate((el) => el.scrollTop))
    .toBeGreaterThan(0);
  expect(
    (await metrics(search)).top,
    "filters should stay in place while rows scroll",
  ).toBe(searchTop);
  expect(await panel.evaluate((el) => el.scrollTop)).toBe(0);
  const main = await metrics(page.locator("main").first());
  expect(main.scrollHeight).toBe(main.height);
}

for (const viewport of [
  { width: 1280, height: 720 },
  { width: 1280, height: 500 },
  { width: 390, height: 600 },
]) {
  test.describe(`Lens scrolling at ${viewport.width}x${viewport.height}`, () => {
    test.use({ storageState: ADMIN_STORAGE_PATH, viewport });

    test("trace rows scroll without moving filters or overflowing the panel", async ({
      page,
    }) => {
      await page.goto("/ui/lens?demo=true");
      await expect(
        page.getByText("Where is order #1042?", { exact: true }),
      ).toBeVisible();
      await expectTableOwnsScroll(page, "Traces");
    });

    test("investigation rows scroll without a second panel scrollbar", async ({
      page,
    }) => {
      await page.goto("/ui/lens?demo=true&tab=investigations");
      await expect(
        page.getByRole("row", { name: "Support quality", exact: true }),
      ).toBeVisible();
      // The desktop demo fits without scrolling; shrinking the viewport exercises overflow.
      await page.setViewportSize({ width: viewport.width, height: 400 });
      await expectTableOwnsScroll(page, "Investigations");
    });
  });
}
