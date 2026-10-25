import { fireEvent, render, screen } from "@testing-library/react";
import { formatDistanceToNow } from "date-fns";
import { zhCN } from "date-fns/locale";
import moment from "moment";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import type { Language } from "@/i18n/language";
import type { LogEntry } from "../types";
import { DrawerHeader } from "./DrawerHeader";

function LanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

function renderHeader(startTime: string, language: Language = "zh-CN") {
  const log = {
    request_id: "request-time-locale",
    model: "Home",
    startTime,
  } as LogEntry;
  render(
    <LanguageProvider defaultLanguage={language} userId="drawer-time" storageScope="drawer-time-test">
      <LanguageSwitch />
      <DrawerHeader
        log={log}
        onClose={vi.fn()}
        onPrevious={vi.fn()}
        onNext={vi.fn()}
        isSidebarCollapsed={false}
        onToggleSidebar={vi.fn()}
        statusLabel="Success"
        statusColor="success"
        environment="default"
      />
    </LanguageProvider>,
  );
  return log;
}

describe("日志详情日期语言", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-10-25T04:00:00+08:00"));
  });

  afterEach(() => vi.useRealTimers());

  it.each(["2026-10-25T09:00:00+08:00", "2026-10-24T23:00:00+08:00"])(
    "中文日期与相对时间可切回英文且保留本地时区和原始时间 %s",
    (startTime) => {
      const localTimestamp = moment(startTime).format("YYYY-MM-DD HH:mm:ss");
      const englishTimestamp = moment(startTime).format("MMM D, YYYY h:mm:ss A");
      const englishRelativeTime = `(${moment(startTime).fromNow()})`;
      const chineseRelativeTime = `(${formatDistanceToNow(new Date(startTime), { addSuffix: true, locale: zhCN })})`;
      const previousGlobalLocale = moment.locale();
      const log = renderHeader(startTime);

      expect(screen.getByText(localTimestamp)).toBeInTheDocument();
      expect(screen.getByText(chineseRelativeTime)).toBeInTheDocument();
      expect(screen.getByText("Home")).toBeInTheDocument();
      expect(moment.locale()).toBe(previousGlobalLocale);

      fireEvent.click(screen.getByRole("button", { name: "English" }));

      expect(screen.getByText(englishTimestamp)).toBeInTheDocument();
      expect(screen.getByText(englishRelativeTime)).toBeInTheDocument();
      expect(log.startTime).toBe(startTime);
    },
  );

  it("无效日期保留原有占位而不让本地化展示抛出异常", () => {
    renderHeader("2026-13-25T09:00:00+08:00");

    expect(screen.getByText("Invalid date")).toBeInTheDocument();
    expect(screen.getByText("(Invalid date)")).toBeInTheDocument();
  });
});
