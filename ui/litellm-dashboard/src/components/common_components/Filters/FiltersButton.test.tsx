import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FiltersButton } from "./FiltersButton";

describe("FiltersButton", () => {
  it("should render", () => {
    const onClick = vi.fn();
    render(<FiltersButton onClick={onClick} active={false} hasActiveFilters={false} />);
    expect(screen.getByRole("button", { name: /filters/i })).toBeInTheDocument();
  });

  it("should call onClick when clicked", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<FiltersButton onClick={onClick} active={false} hasActiveFilters={false} />);

    const button = screen.getByRole("button", { name: /filters/i });
    await user.click(button);

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("should show badge when hasActiveFilters is true", () => {
    const onClick = vi.fn();
    const { container } = render(<FiltersButton onClick={onClick} active={false} hasActiveFilters={true} />);
    const button = screen.getByRole("button", { name: /filters/i });
    const badgeWrapper = button.closest(".ant-badge");
    expect(badgeWrapper).toBeInTheDocument();
  });

  it("should render custom label when provided", () => {
    const onClick = vi.fn();
    render(<FiltersButton onClick={onClick} active={false} hasActiveFilters={false} label="Advanced Filters" />);
    expect(screen.getByRole("button", { name: /advanced filters/i })).toBeInTheDocument();
  });
});
