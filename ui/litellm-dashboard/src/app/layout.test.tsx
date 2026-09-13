import fs from "node:fs";
import path from "node:path";

import { describe, expect, it, vi } from "vitest";

vi.mock("./globals.css", () => ({}));
vi.mock("next/font/google", () => ({
  Inter: () => ({ className: "inter" }),
}));

const { metadata } = await import("./layout");

const appDir = path.dirname(new URL(import.meta.url).pathname);
const dashboardDir = path.resolve(appDir, "..", "..");

describe("root layout favicon", () => {
  it("points the only declared icon at the proxy's /get_favicon endpoint", () => {
    expect(metadata.icons).toEqual({ icon: "/get_favicon" });
  });

  it("keeps favicon.ico out of the app directory so Next does not emit a competing icon link", () => {
    expect(fs.existsSync(path.join(appDir, "favicon.ico"))).toBe(false);
  });

  it("ships the default favicon as a public asset so the exported build still serves favicon.ico", () => {
    expect(fs.existsSync(path.join(dashboardDir, "public", "favicon.ico"))).toBe(true);
  });
});
