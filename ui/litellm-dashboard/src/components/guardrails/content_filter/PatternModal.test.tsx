import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import PatternModal from "./PatternModal";

describe("PatternModal", () => {
  const mockOnAdd = vi.fn();
  const mockOnCancel = vi.fn();
  const mockOnPatternNameChange = vi.fn();
  const mockOnActionChange = vi.fn();

  const mockPrebuiltPatterns = [
    {
      name: "us_ssn",
      display_name: "US SSN",
      category: "PII Patterns",
      description: "US Social Security Number",
    },
    {
      name: "email",
      display_name: "Email",
      category: "PII Patterns",
      description: "Email addresses",
    },
    {
      name: "visa",
      display_name: "Visa",
      category: "Financial Patterns",
      description: "Visa credit card numbers",
    },
    {
      name: "aws_access_key",
      display_name: "AWS Access Key",
      category: "Credential Patterns",
      description: "AWS Access Keys",
    },
  ];

  const mockCategories = [
    "PII Patterns",
    "Financial Patterns",
    "Credential Patterns",
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the pattern modal dialog with a pattern-type combobox when visible", async () => {
    render(
      <PatternModal
        visible={true}
        prebuiltPatterns={mockPrebuiltPatterns}
        categories={mockCategories}
        selectedPatternName=""
        patternAction="BLOCK"
        onPatternNameChange={mockOnPatternNameChange}
        onActionChange={mockOnActionChange}
        onAdd={mockOnAdd}
        onCancel={mockOnCancel}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Add prebuilt pattern")).toBeInTheDocument();
    });

    // Two comboboxes should be present: pattern type + action.
    const comboboxes = screen.getAllByRole("combobox");
    expect(comboboxes.length).toBeGreaterThanOrEqual(2);

    // The footer action buttons wire to the expected handlers.
    screen.getByRole("button", { name: "Cancel" }).click();
    expect(mockOnCancel).toHaveBeenCalled();

    screen.getByRole("button", { name: "Add" }).click();
    expect(mockOnAdd).toHaveBeenCalled();
  });
});
