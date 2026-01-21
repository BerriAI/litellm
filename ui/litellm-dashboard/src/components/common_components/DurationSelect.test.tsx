import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
  });

  it("should apply className prop", () => {
    render(<DurationSelect className="test-class" />);
    const select = screen.getByRole("combobox");
    expect(select.closest(".test-class")).toBeInTheDocument();
  });

  it("should call onChange when an option is selected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<DurationSelect onChange={onChange} />);

    const select = screen.getByRole("combobox");
    await user.click(select);

    const dailyOption = screen.getByText("Daily");
    await user.click(dailyOption);

    expect(onChange).toHaveBeenCalledWith("24h", expect.any(Object));
  });

  it("should accept and pass value prop to Select", () => {
    render(<DurationSelect value="7d" />);
    const select = screen.getByRole("combobox");
    expect(select).toBeInTheDocument();
  });
});
