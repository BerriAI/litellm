import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ResetFiltersButton } from "./ResetFiltersButton";

describe("ResetFiltersButton", () => {
  it("should render", () => {
    const onClick = vi.fn();
    render(<ResetFiltersButton onClick={onClick} />);
    expect(screen.getByRole("button", { name: /reset filters/i })).toBeInTheDocument();
  });

  it("should call onClick when clicked", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<ResetFiltersButton onClick={onClick} />);

    const button = screen.getByRole("button", { name: /reset filters/i });
    await user.click(button);

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("should render custom label when provided", () => {
    const onClick = vi.fn();
    render(<ResetFiltersButton onClick={onClick} label="Clear All" />);
    expect(screen.getByRole("button", { name: /clear all/i })).toBeInTheDocument();
  });
});
