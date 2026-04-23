import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UsageViewSelect } from "./UsageViewSelect";

describe("UsageViewSelect", () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  it("should render", () => {
    render(
      <UsageViewSelect
        value="global"
        onChange={mockOnChange}
        isAdmin={false}
      />,
    );

    expect(screen.getByText("Usage View")).toBeInTheDocument();
    expect(
      screen.getByText("Select the usage data you want to view"),
    ).toBeInTheDocument();
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should render the combobox with the currently-selected option label", () => {
    /**
     * The shadcn Select renders as a Radix trigger whose displayed value
     * follows the selected SelectItem. Driving the dropdown open in JSDOM
     * isn't reliable (Radix pointer-capture), so we assert trigger presence
     * + that the 'Your Usage' label for non-admin 'global' renders inside
     * the combobox content. Option-level interaction is covered by
     * Playwright.
     */
    render(
      <UsageViewSelect
        value="global"
        onChange={mockOnChange}
        isAdmin={false}
      />,
    );
    const trigger = screen.getByRole("combobox");
    expect(trigger).toBeInTheDocument();
    // In non-admin mode the global label is 'Your Usage'
    expect(trigger).toHaveTextContent(/Your Usage/);
  });
});
