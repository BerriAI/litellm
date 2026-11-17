import { afterEach, describe, expect, it, vi } from "vitest";

import { downloadBlob, exportFilename } from "./utils";

describe("exportFilename", () => {
  it("composes entity, export type, range and format", () => {
    const name = exportFilename("team", "daily_with_keys", "csv", {
      from: new Date(2025, 0, 5),
      to: new Date(2025, 0, 31),
    });

    expect(name).toBe("team_usage_daily_with_keys_2025-01-05_2025-01-31.csv");
  });

  it("falls back to 'all' when a bound is missing", () => {
    expect(exportFilename("user", "daily_with_users", "json", { from: undefined, to: undefined })).toBe(
      "user_usage_daily_with_users_all_all.json",
    );
  });
});

describe("downloadBlob", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("triggers an anchor click carrying the filename and revokes the object URL", () => {
    const click = vi.fn();
    const revoke = vi.fn();
    const anchor = { href: "", download: "", click, remove: vi.fn() };
    const createElement = vi.fn(() => anchor);
    const appendChild = vi.fn((node: unknown) => node);
    const removeChild = vi.fn((node: unknown) => node);
    const fakeURL = { ...URL, createObjectURL: vi.fn(() => "blob:mock"), revokeObjectURL: revoke };
    vi.stubGlobal("URL", fakeURL);
    vi.stubGlobal("window", { URL: fakeURL });
    vi.stubGlobal("document", {
      createElement,
      body: { appendChild, removeChild },
    });

    downloadBlob(new Blob(["data"]), "team_usage_daily_2025-01-01_2025-01-31.csv");

    expect(createElement).toHaveBeenCalledWith("a");
    expect(anchor.download).toBe("team_usage_daily_2025-01-01_2025-01-31.csv");
    expect(anchor.href).toBe("blob:mock");
    expect(click).toHaveBeenCalledOnce();
    expect(revoke).toHaveBeenCalledWith("blob:mock");
  });
});
