import { describe, expect, it } from "vitest";
import { displayDate, displayTime, displayTimestamp } from "./displayDate";

describe("dashboard date display", () => {
  const date = new Date(2026, 11, 12, 13, 45, 23);

  it("formats dates and clocks in Chinese regardless of the browser locale", () => {
    expect(displayDate(date, "zh-CN")).toBe("2026年12月12日");
    expect(displayTime(date, "zh-CN")).toBe("13:45");
    expect(displayTimestamp(date, "zh-CN")).toBe("2026年12月12日 13:45:23");
  });

  it("preserves the old browser-default presentation for English", () => {
    const timeOptions = { hour: "2-digit", minute: "2-digit" } as const;
    expect(displayDate(date, "en")).toBe(date.toLocaleDateString());
    expect(displayTime(date, "en")).toBe(date.toLocaleTimeString([], timeOptions));
    expect(displayTimestamp(date, "en")).toBe(date.toLocaleString());
  });
});
