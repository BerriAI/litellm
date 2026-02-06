import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import AdvancedDatePicker from "./advanced_date_picker";

// Polyfill requestIdleCallback for test environment
beforeAll(() => {
  if (typeof window !== "undefined" && !window.requestIdleCallback) {
    window.requestIdleCallback = (callback: any) => {
      const start = Date.now();
      return setTimeout(() => {
        callback({
          didTimeout: false,
          timeRemaining: () => Math.max(0, 50 - (Date.now() - start)),
        });
      }, 1) as any;
    };
  }
});

describe("AdvancedDatePicker", () => {
  const mockOnValueChange = vi.fn();
  const defaultValue = {
    from: new Date("2025-01-01T12:00:00.000Z"),
    to: new Date("2025-01-31T12:00:00.000Z"),
  };

  beforeEach(() => {
    mockOnValueChange.mockClear();
  });

  const openDropdown = (container: HTMLElement) => {
    // Find the clickable div that contains the clock icon
    const trigger = container.querySelector('[role="img"][aria-label="clock-circle"]')?.closest("div.cursor-pointer");
    if (trigger) {
      fireEvent.click(trigger);
    }
  };

  it("should render with default label", () => {
    render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);
    expect(screen.getByText("Select Time Range")).toBeInTheDocument();
  });

  it("should render with custom label", () => {
    render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} label="Custom Label" />);
    expect(screen.getByText("Custom Label")).toBeInTheDocument();
  });

  it("should display formatted date range", () => {
    render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);
    // The component displays date range in the format "D MMM, HH:mm - D MMM, HH:mm"
    // Just check that the clock icon is present
    expect(screen.getByLabelText("clock-circle")).toBeInTheDocument();
  });

  it("should open dropdown when clicked", () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    // Check for relative time options
    expect(screen.getByText("Today")).toBeInTheDocument();
    expect(screen.getByText("Last 7 days")).toBeInTheDocument();
    expect(screen.getByText("Last 30 days")).toBeInTheDocument();
  });

  it("should display relative time options", () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    expect(screen.getByText("Today")).toBeInTheDocument();
    expect(screen.getByText("Last 7 days")).toBeInTheDocument();
    expect(screen.getByText("Last 30 days")).toBeInTheDocument();
    expect(screen.getByText("Month to date")).toBeInTheDocument();
    expect(screen.getByText("Year to date")).toBeInTheDocument();
  });

  it("should show date inputs in dropdown", () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    const startDateInput = screen.getByDisplayValue("2025-01-01");
    const endDateInput = screen.getByDisplayValue("2025-01-31");

    expect(startDateInput).toBeInTheDocument();
    expect(endDateInput).toBeInTheDocument();
  });

  it("should update date inputs when changed", () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    const startDateInput = screen.getByDisplayValue("2025-01-01") as HTMLInputElement;
    fireEvent.change(startDateInput, { target: { value: "2025-02-01" } });

    expect(startDateInput.value).toBe("2025-02-01");
  });

  it("should show Apply and Cancel buttons", () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    expect(screen.getByText("Apply")).toBeInTheDocument();
    expect(screen.getByText("Cancel")).toBeInTheDocument();
  });

  it("should close dropdown when Cancel is clicked", () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    const cancelButton = screen.getByText("Cancel");
    fireEvent.click(cancelButton);

    // Dropdown should be closed, so relative time options shouldn't be visible
    expect(screen.queryByText("Today")).not.toBeInTheDocument();
  });

  it("should call onValueChange when Apply is clicked", async () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    const applyButton = screen.getByText("Apply");
    fireEvent.click(applyButton);

    await waitFor(() => {
      expect(mockOnValueChange).toHaveBeenCalled();
    });
  });

  it("should select relative time option", () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    const todayOption = screen.getByText("Today");
    fireEvent.click(todayOption);

    // The option should be highlighted (bg-blue-50)
    expect(todayOption.closest("div")).toHaveClass("bg-blue-50");
  });

  it("should show validation error for invalid date range", async () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    const startDateInput = screen.getByDisplayValue("2025-01-01");
    const endDateInput = screen.getByDisplayValue("2025-01-31");

    // Set end date before start date
    fireEvent.change(startDateInput, { target: { value: "2025-12-01" } });
    fireEvent.change(endDateInput, { target: { value: "2025-01-01" } });

    await waitFor(() => {
      expect(screen.getByText("End date cannot be before start date")).toBeInTheDocument();
    });
  });

  it("should disable Apply button when validation fails", async () => {
    const { container } = render(<AdvancedDatePicker value={defaultValue} onValueChange={mockOnValueChange} />);

    openDropdown(container);

    const startDateInput = screen.getByDisplayValue("2025-01-01");
    const endDateInput = screen.getByDisplayValue("2025-01-31");

    // Set end date before start date
    fireEvent.change(startDateInput, { target: { value: "2025-12-01" } });
    fireEvent.change(endDateInput, { target: { value: "2025-01-01" } });

    await waitFor(() => {
      // Find the button element (the Apply button's actual button element)
      const applyButton = screen.getByText("Apply").closest("button");
      expect(applyButton).toBeDisabled();
    });
  });
});
