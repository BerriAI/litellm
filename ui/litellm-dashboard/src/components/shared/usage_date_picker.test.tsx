import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";
import UsageDatePicker from "./usage_date_picker";

beforeAll(() => {
  vi.stubGlobal("requestIdleCallback", (cb: IdleRequestCallback) => {
    cb({ didTimeout: false, timeRemaining: () => 50 } as IdleDeadline);
    return 0;
  });
});

describe("UsageDatePicker (tremor DateRangePicker on date-fns 4)", () => {
  const value = { from: new Date(2026, 5, 1), to: new Date(2026, 5, 15) };

  it("renders the formatted range label", () => {
    render(<UsageDatePicker value={value} onValueChange={() => {}} />);

    const triggerText = screen.getAllByRole("button")[0].textContent ?? "";
    expect(triggerText).toMatch(/Jun/);
    expect(triggerText).toMatch(/2026/);
    expect(triggerText).toMatch(/15/);
  });

  it("opens the calendar and renders a full month grid", async () => {
    const user = userEvent.setup();
    render(<UsageDatePicker value={value} onValueChange={() => {}} />);

    await user.click(screen.getAllByRole("button")[0]);

    const grid = await screen.findByRole("grid");
    const dayCells = within(grid).getAllByRole("gridcell");
    expect(dayCells.length).toBeGreaterThanOrEqual(28);
    expect(within(grid).getByText("15")).toBeInTheDocument();
  });

  it("fires onValueChange when a day is selected", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(<UsageDatePicker value={value} onValueChange={onValueChange} />);

    await user.click(screen.getAllByRole("button")[0]);
    const grid = await screen.findByRole("grid");
    await user.click(within(grid).getByText("10"));

    expect(onValueChange).toHaveBeenCalled();
    const newValue = onValueChange.mock.calls[0][0];
    expect(newValue.from).toBeInstanceOf(Date);
  });
});
