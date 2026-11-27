import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { I18nextProvider } from "react-i18next";

import i18n from "@/i18n/i18n";
import zh from "@/i18n/locales/zh-CN.json";
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
  it("includes year, 24h time and the IANA timezone", () => {
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(formatFullTimestamp(new Date(2026, 6, 7, 9, 50, 13))).toBe(`Jul 7, 2026, 09:50:13 (${timeZone})`);
  });
});

describe("DateCell", () => {
  it("uses Chinese date order and a 24-hour clock in the cell and its tooltip", async () => {
    const locale = i18n.cloneInstance({ forkResourceStore: true });
    locale.addResourceBundle("zh-CN", "translation", zh);
    await locale.changeLanguage("zh-CN");
    const value = "2026-11-27T03:23:23Z";
    render(
      <I18nextProvider i18n={locale}>
        <DateCell value={value} timeZone="UTC" />
        <DateCell value={value} timeZone="UTC" precision="date" />
      </I18nextProvider>,
    );

    const timestamp = screen.getByText("11月27日 03:23:23 UTC");
    expect(screen.getByText("2026年11月27日 UTC")).toBeInTheDocument();
    await userEvent.hover(timestamp);
    expect(await screen.findByText("2026年11月27日 03:23:23 (UTC)")).toBeInTheDocument();
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

describe("DateCell with timeZone UTC", () => {
  const originalTimezone: string | undefined = process.env.TZ;

  afterEach(() => {
    if (originalTimezone === undefined) {
      delete process.env.TZ;
      return;
    }
    process.env.TZ = originalTimezone;
  });

  it("prints the UTC clock time and labels it UTC for a viewer in another zone", async () => {
    process.env.TZ = "America/Los_Angeles";
    const user = userEvent.setup();
    render(<DateCell value="2026-11-01T00:00:00Z" timeZone="UTC" />);

    const cell = screen.getByText("Nov 1, 00:00:00 UTC");
    await user.hover(cell);
    expect(await screen.findByText("Nov 1, 2026, 00:00:00 (UTC)")).toBeInTheDocument();
  });

  it("labels date precision with UTC too", () => {
    process.env.TZ = "Asia/Tokyo";
    expect(formatCellDate(new Date("2026-10-31T20:00:00Z"), "date", "UTC")).toBe("Oct 31, 2026 UTC");
  });

  it("keeps the local default unlabeled for every other column", () => {
    process.env.TZ = "America/Los_Angeles";
    expect(formatCellDate(new Date("2026-11-01T00:00:00Z"), "datetime")).toBe("Oct 31, 17:00:00");
  });
});
