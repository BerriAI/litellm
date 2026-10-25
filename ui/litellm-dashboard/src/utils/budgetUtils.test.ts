import { afterEach, describe, expect, it } from "vitest";

import { describeBudgetResetSchedule, formatBudgetReset, formatBudgetResetUtc } from "./budgetUtils";
import { translate, type Translate } from "@/i18n/language";

const originalTimezone: string | undefined = process.env.TZ;

afterEach(() => {
  if (originalTimezone === undefined) {
    delete process.env.TZ;
    return;
  }
  process.env.TZ = originalTimezone;
});

describe("formatBudgetResetUtc", () => {
  it("formats the reset date in UTC regardless of the local timezone", () => {
    process.env.TZ = "America/Los_Angeles";
    const resetAt: string = "2026-10-20T00:00:00Z";

    expect(formatBudgetResetUtc(resetAt)).toBe("Oct 20, 2026");
    expect(formatBudgetReset(resetAt)).toBe("Oct 19, 2026");
  });
});

describe("describeBudgetResetSchedule", () => {
  const chinese: Translate = (message, values) => translate("zh-CN", message, values);

  it("汉化未安排和无周期的重置说明，保留原始周期值", () => {
    expect(describeBudgetResetSchedule(null, null, chinese).label).toBe("不重置");
    expect(describeBudgetResetSchedule(null, null, chinese).tooltip).toBe(
      "此预算没有重置周期，当前周期费用将在成员关系存续期间持续累计。",
    );
    const unknown = describeBudgetResetSchedule("Home", "invalid", chinese);
    expect(unknown.label).toBe("未知");
    expect(unknown.resetAt).toBeNull();
    expect(unknown.tooltip).toBe("每 Home 重置，但尚未安排下次重置时间。重置时超出结转上限的费用将计入下一周期。");
  });

  it.each(["30d", null])("汉化已安排的重置说明，保留日期、时区和周期 %s", (duration) => {
    const resetAt = new Date(2026, 6, 15, 12, 0, 0).toISOString();
    const english = describeBudgetResetSchedule(duration, resetAt);
    const schedule = describeBudgetResetSchedule(duration, resetAt, chinese);
    expect(schedule.resetAt).toBe(english.resetAt);
    expect(schedule.label).toBe(english.label);
    expect(schedule.tooltip).toBe(
      `当前周期费用${duration ? "每 30d" : "按计划"}重置。下次重置为 ${english.label}（${Intl.DateTimeFormat().resolvedOptions().timeZone}）。重置时超出结转上限的费用将计入下一周期。`,
    );
  });

  it("reports Never resets when the budget has no duration and no scheduled reset", () => {
    const schedule = describeBudgetResetSchedule(null, null);
    expect(schedule.label).toBe("Never resets");
    expect(schedule.resetAt).toBeNull();
  });

  it("reports Unknown when a duration is set but no reset has been scheduled yet", () => {
    const schedule = describeBudgetResetSchedule("30d", null);
    expect(schedule.label).toBe("Unknown");
    expect(schedule.tooltip).toContain("every 30d");
  });

  it("shows the next reset with its time zone and the rollover carryover note", () => {
    const resetAt = new Date(2026, 6, 15, 12, 0, 0);
    const schedule = describeBudgetResetSchedule("30d", resetAt.toISOString());
    expect(schedule.label).toBe("Jul 15, 2026 12:00");
    expect(schedule.resetAt).toBe(resetAt.toISOString());
    expect(schedule.tooltip).toContain(Intl.DateTimeFormat().resolvedOptions().timeZone);
    expect(schedule.tooltip).toContain("carries into the next cycle");
  });
});
