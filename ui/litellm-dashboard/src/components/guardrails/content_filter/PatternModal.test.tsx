import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PatternModal from "./PatternModal";

describe("PatternModal", () => {
  const mockOnAdd = vi.fn();
  const mockOnCancel = vi.fn();
  const mockOnPatternNameChange = vi.fn();
  const mockOnActionChange = vi.fn();

  const mockPrebuiltPatterns = [
    { name: "us_ssn", category: "PII Patterns", description: "US Social Security Number" },
    { name: "email", category: "PII Patterns", description: "Email addresses" },
    { name: "visa", category: "Financial Patterns", description: "Visa credit card numbers" },
    { name: "aws_access_key", category: "Credential Patterns", description: "AWS Access Keys" },
  ];

  const mockCategories = ["PII Patterns", "Financial Patterns", "Credential Patterns"];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should show dropdown with prebuilt pattern options grouped by category", async () => {
    /**
     * Tests that the modal displays a dropdown with prebuilt patterns
     * organized by category. This verifies the pattern selection UI is working.
     */
    const user = userEvent.setup();

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
      />
    );

    // Wait for modal to be visible
    await waitFor(() => {
      expect(screen.getByText("Add prebuilt pattern")).toBeInTheDocument();
    });

    // Find the pattern type dropdown by looking for the first combobox input
    const comboboxes = screen.getAllByRole("combobox");
    const dropdown = comboboxes[0]; // First combobox is the pattern selector
    expect(dropdown).toBeInTheDocument();

    // Click to open the dropdown
    await user.click(dropdown);

    // Verify that pattern options are available in the dropdown
    // Ant Design renders Select options in a portal, so we need to query the whole document
    await waitFor(() => {
      const options = document.querySelectorAll('.ant-select-item-option');
      expect(options.length).toBeGreaterThan(0);
    });

    // Verify categories are shown as group labels
    await waitFor(() => {
      expect(document.body).toHaveTextContent("PII Patterns");
      expect(document.body).toHaveTextContent("Financial Patterns");
      expect(document.body).toHaveTextContent("Credential Patterns");
    });

    // Verify pattern options are available
    expect(document.body).toHaveTextContent("us_ssn");
    expect(document.body).toHaveTextContent("email");
    expect(document.body).toHaveTextContent("visa");
    expect(document.body).toHaveTextContent("aws_access_key");

    // Select a pattern by clicking on its option element
    const ssnOption = Array.from(document.querySelectorAll('.ant-select-item-option')).find(
      el => el.textContent === "us_ssn"
    ) as HTMLElement;
    await user.click(ssnOption);

    // Verify the change handler was called with the pattern name
    // Note: Ant Design Select calls onChange with (value, option), so we check if it was called
    expect(mockOnPatternNameChange).toHaveBeenCalled();
    const callArgs = mockOnPatternNameChange.mock.calls[0];
    expect(callArgs[0]).toBe("us_ssn");
  });
});

