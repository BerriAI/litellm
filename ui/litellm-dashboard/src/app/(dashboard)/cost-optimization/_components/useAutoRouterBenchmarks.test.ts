import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ formatDate: vi.fn() }));
vi.mock("@/lib/http/api", () => ({ $api: { useQuery: vi.fn() } }));

import { benchmarksWindow } from "./useAutoRouterBenchmarks";

const localDay =
  (offsetHours: number) =>
  (d: Date): string =>
    new Date(d.getTime() + offsetHours * 3_600_000).toISOString().slice(0, 10);

const pacific = localDay(-7);
const tokyo = localDay(+9);

describe("benchmarksWindow", () => {
  it("passes a historical range through as the picked local calendar days", () => {
    const now = new Date("2026-08-21T19:00:00Z");
    const range = { from: new Date("2026-07-06T19:00:00Z"), to: new Date("2026-08-05T19:00:00Z") };
    expect(benchmarksWindow(range, now, pacific)).toEqual({ start_date: "2026-07-06", end_date: "2026-08-05" });
  });

  it("keeps a range ending today unchanged while the local and UTC days still agree", () => {
    const now = new Date("2026-08-21T19:00:00Z");
    const range = { from: new Date("2026-07-22T19:00:00Z"), to: now };
    expect(benchmarksWindow(range, now, pacific)).toEqual({ start_date: "2026-07-22", end_date: "2026-08-21" });
  });

  it("extends a range ending today to the current UTC day once UTC rolls past local midnight", () => {
    const now = new Date("2026-08-22T03:00:00Z");
    const range = { from: new Date("2026-07-22T19:00:00Z"), to: now };
    expect(benchmarksWindow(range, now, pacific)).toEqual({ start_date: "2026-07-22", end_date: "2026-08-22" });
  });

  it("never shrinks a range for a caller east of UTC whose local day is already tomorrow", () => {
    const now = new Date("2026-08-21T19:00:00Z");
    const range = { from: new Date("2026-07-22T19:00:00Z"), to: now };
    expect(benchmarksWindow(range, now, tokyo)).toEqual({ start_date: "2026-07-23", end_date: "2026-08-22" });
  });

  it("sends no dates while the picker is missing either end", () => {
    const now = new Date("2026-08-21T19:00:00Z");
    expect(benchmarksWindow({ from: now }, now, pacific)).toEqual({});
    expect(benchmarksWindow({ to: now }, now, pacific)).toEqual({});
  });
});
