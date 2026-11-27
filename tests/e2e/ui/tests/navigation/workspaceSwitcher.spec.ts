import test, { expect } from "@playwright/test";
import type { Locator } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { dismissFeedbackPopup } from "../../helpers/navigation";

const WCAG_AA_TEXT_CONTRAST = 4.5;

async function textContrast(element: Locator): Promise<number> {
  return element.evaluate(async (el) => {
    const item = el.closest('[role="menuitem"]') ?? el;
    await Promise.all(item.getAnimations({ subtree: true }).map((animation) => animation.finished));
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 1;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    const toRgba = (color: string): [number, number, number, number] => {
      ctx.clearRect(0, 0, 1, 1);
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, 1, 1);
      const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
      return [r, g, b, a / 255];
    };
    const over = (top: number[], bottom: number[]) =>
      top.map((c, i) => (i < 3 ? c * top[3] + bottom[i] * (1 - top[3]) : 1));
    const layers = [];
    for (let node: Element | null = el; node; node = node.parentElement) {
      layers.push(toRgba(getComputedStyle(node).backgroundColor));
    }
    const background = layers.reduceRight<number[]>((below, layer) => over(layer, below), [255, 255, 255, 1]);
    const text = over(toRgba(getComputedStyle(el).color), background);
    const luminance = (rgb: number[]) => {
      const [r, g, b] = rgb.slice(0, 3).map((v) => {
        const c = v / 255;
        return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const [hi, lo] = [luminance(text), luminance(background)].sort((a, b) => b - a);
    return (hi + 0.05) / (lo + 0.05);
  });
}

test.describe("workspace switcher", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("keeps the Beta badge readable while its menu item is highlighted", async ({ page }) => {
    await page.goto("/ui");
    await dismissFeedbackPopup(page);

    await page
      .getByRole("button", { name: /^Switch workspace, current:/ })
      .first()
      .click();
    const lensItem = page.getByRole("menuitem", { name: /^Lens\b/ });
    const badge = lensItem.getByText("Beta", { exact: true });
    await expect(badge).toBeVisible();

    await lensItem.hover();
    await expect(lensItem).toHaveAttribute("data-highlighted", "");

    expect(await textContrast(badge)).toBeGreaterThanOrEqual(WCAG_AA_TEXT_CONTRAST);
  });
});
