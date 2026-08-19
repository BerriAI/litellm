import { render, screen } from "@testing-library/react";
import { useState } from "react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import DurationSelect from "./DurationSelect";

describe("DurationSelect", () => {
  it("should render", () => {
    render(<DurationSelect />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should render all three duration options", async () => {
    const user = userEvent.setup();
    render(<DurationSelect />);

    const select = screen.getByRole("combobox");
    await user.click(select);

    expect(screen.getByText("Daily")).toBeInTheDocument();
    expect(screen.getByText("Weekly")).toBeInTheDocument();
    expect(screen.getByText("Monthly")).toBeInTheDocument();
    const dailyLabel = screen.getByText("Daily");
    const dailyOption = dailyLabel.closest('[role="option"]') ?? dailyLabel;
    await user.click(dailyOption);
  });

  it("should apply className prop", () => {
    render(<DurationSelect className="test-class" />);
    const select = screen.getByRole("combobox");
    expect(select.closest(".test-class")).toBeInTheDocument();
  });

  it("should call onChange when an option is selected", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    const onChange = vi.fn();
    render(<DurationSelect onChange={onChange} />);

    const select = screen.getByRole("combobox");
    await user.click(select);

    const dailyLabel = screen.getByText("Daily");
    const dailyOption = dailyLabel.closest('[role="option"]') ?? dailyLabel;
    await user.click(dailyOption);

    expect(onChange).toHaveBeenCalledWith("24h", expect.any(Object));
  });

  it("should accept and pass value prop to Select", () => {
    render(<DurationSelect value="7d" />);
    const select = screen.getByRole("combobox");
    expect(select).toBeInTheDocument();
  });

  it.each([
    ["24h", "Daily"],
    ["7d", "Weekly"],
    ["30d", "Monthly"],
  ])("shows the human label on the trigger for %s", (value, label) => {
    render(<DurationSelect value={value} />);

    expect(screen.getByRole("combobox")).toHaveTextContent(label);
  });

  it("shows the human label on the trigger after the user picks an option", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    const Harness = () => {
      const [value, setValue] = useState("24h");
      return <DurationSelect value={value} onChange={setValue} />;
    };
    render(<Harness />);

    await user.click(screen.getByRole("combobox"));
    const monthly = screen.getByText("Monthly");
    await user.click(monthly.closest('[role="option"]') ?? monthly);

    expect(screen.getByRole("combobox")).toHaveTextContent("Monthly");
  });
});
