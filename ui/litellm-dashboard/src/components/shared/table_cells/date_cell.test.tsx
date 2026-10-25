import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";

import { DateCell, formatCellDate, formatFullTimestamp } from "./date_cell";

const localIso = new Date(2026, 6, 7, 9, 50, 13).toISOString();

describe("formatCellDate", () => {
  it("formats datetime precision as 'MMM D, HH:mm:ss' without a year", () => {
    expect(formatCellDate(new Date(2026, 6, 7, 9, 50, 13), "datetime")).toBe("Jul 7, 09:50:13");
  });

  it("zero-pads hours, minutes and seconds", () => {
    expect(formatCellDate(new Date(2026, 0, 2, 1, 2, 3), "datetime")).toBe("Jan 2, 01:02:03");
  });

  it("formats date precision as 'MMM D, YYYY' with no time", () => {
    expect(formatCellDate(new Date(2026, 11, 31, 23, 59, 59), "date")).toBe("Dec 31, 2026");
  });
});

describe("formatFullTimestamp", () => {
  it("中文完整时间戳保留本地小时和 IANA 时区", () => {
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(formatFullTimestamp(new Date(2026, 6, 7, 9, 50, 13), "zh-CN")).toBe(`2026-07-07 09:50:13 (${timeZone})`);
  });

  it("includes year, 24h time and the IANA timezone", () => {
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(formatFullTimestamp(new Date(2026, 6, 7, 9, 50, 13))).toBe(`Jul 7, 2026, 09:50:13 (${timeZone})`);
  });
});

describe("DateCell", () => {
  it("中文日期随语言切换，保留原有本地时区和英文格式", () => {
    function LanguageSwitch() {
      const { setPreference } = useLanguage();
      return <button onClick={() => setPreference("en")}>English</button>;
    }
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="date" storageScope="date-language">
        <LanguageSwitch />
        <DateCell value={localIso} />
        <DateCell value={localIso} precision="date" />
      </LanguageProvider>,
    );
    expect(screen.getByText("2026-07-07 09:50:13")).toBeInTheDocument();
    expect(screen.getByText("2026-07-07")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByText("Jul 7, 09:50:13")).toBeInTheDocument();
    expect(screen.getByText("Jul 7, 2026")).toBeInTheDocument();
  });

  it("renders the datetime format by default", () => {
    render(<DateCell value={localIso} />);
    expect(screen.getByText("Jul 7, 09:50:13")).toBeInTheDocument();
  });

  it("renders date-only when precision is 'date'", () => {
    render(<DateCell value={localIso} precision="date" />);
    expect(screen.getByText("Jul 7, 2026")).toBeInTheDocument();
  });

  it("renders '-' for null and undefined", () => {
    const { rerender } = render(<DateCell value={null} />);
    expect(screen.getByText("-")).toBeInTheDocument();
    rerender(<DateCell value={undefined} />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("renders the custom fallback for empty values", () => {
    render(<DateCell value="" fallback="Never" />);
    expect(screen.getByText("Never")).toBeInTheDocument();
  });

  it("renders the fallback instead of 'Invalid Date' for unparseable input", () => {
    render(<DateCell value="not-a-date" fallback="Unknown" />);
    expect(screen.getByText("Unknown")).toBeInTheDocument();
    expect(screen.queryByText(/Invalid/)).not.toBeInTheDocument();
  });
});
