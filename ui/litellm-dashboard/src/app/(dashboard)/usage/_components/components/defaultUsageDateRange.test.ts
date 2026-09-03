import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveDefaultUsageDateRange } from "./defaultUsageDateRange";

const local = (y: number, m: number, d: number, time = "00:00:00.000") =>
  new Date(`${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}T${time}`);
const END_OF_DAY = "23:59:59.999";

describe("resolveDefaultUsageDateRange", () => {
  describe("fallback when the admin has not configured a default", () => {
    it.each([undefined, null, "", "last_90_days", 7, { id: "month_to_date" }])(
      "opens on the trailing 7 days for %j",
      (setting) => {
        const now = local(2026, 3, 15, "10:30:00.000");

        const range = resolveDefaultUsageDateRange(setting, now);

        expect(range.from).toEqual(local(2026, 3, 8, "10:30:00.000"));
        expect(range.to).toEqual(now);
      },
    );
  });

  describe("month_to_date", () => {
    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it("starts at local midnight on the 1st of the current month and ends at the end of today", () => {
      const range = resolveDefaultUsageDateRange("month_to_date", local(2026, 3, 15, "10:30:00.000"));

      expect(range.from).toEqual(local(2026, 3, 1));
      expect(range.to).toEqual(local(2026, 3, 15, END_OF_DAY));
    });

    it("rolls over to the new month as soon as the 1st begins", () => {
      const lastOfMarch = resolveDefaultUsageDateRange("month_to_date", local(2026, 3, 31, END_OF_DAY));
      const firstOfApril = resolveDefaultUsageDateRange("month_to_date", local(2026, 4, 1, "00:00:00.001"));

      expect(lastOfMarch.from).toEqual(local(2026, 3, 1));
      expect(firstOfApril.from).toEqual(local(2026, 4, 1));
      expect(firstOfApril.to).toEqual(local(2026, 4, 1, END_OF_DAY));
    });

    it("keeps the month boundary in the viewer's local timezone rather than UTC", () => {
      vi.stubEnv("TZ", "Pacific/Kiritimati");

      const range = resolveDefaultUsageDateRange("month_to_date", new Date("2026-01-31T10:30:00.000Z"));

      expect(range.from.toISOString()).toBe("2026-01-31T10:00:00.000Z");
      expect(range.to.toISOString()).toBe("2026-02-01T09:59:59.999Z");
    });
  });

  it.each([
    ["today", local(2026, 3, 15), local(2026, 3, 15, END_OF_DAY)],
    ["last_7_days", local(2026, 3, 8), local(2026, 3, 15, END_OF_DAY)],
    ["last_30_days", local(2026, 2, 13), local(2026, 3, 15, END_OF_DAY)],
    ["year_to_date", local(2026, 1, 1), local(2026, 3, 15, END_OF_DAY)],
  ])("resolves %s to the same window the date picker preset uses", (preset, from, to) => {
    const range = resolveDefaultUsageDateRange(preset, local(2026, 3, 15, "10:30:00.000"));

    expect(range.from).toEqual(from);
    expect(range.to).toEqual(to);
  });
});
