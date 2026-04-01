import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CustomPatternModal from "./CustomPatternModal";

describe("CustomPatternModal", () => {
  const mockOnAdd = vi.fn();
  const mockOnCancel = vi.fn();
  const mockOnNameChange = vi.fn();
  const mockOnRegexChange = vi.fn();
  const mockOnActionChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should allow entering regex and pattern name and call onAdd when clicking add button", async () => {
    /**
     * Tests that user can enter a pattern name and regex, then clicking Add
     * calls the onAdd callback. This is the core functionality of adding custom patterns.
     */
    const user = userEvent.setup();

    render(
      <CustomPatternModal
        visible={true}
        patternName=""
        patternRegex=""
        patternAction="BLOCK"
        onNameChange={mockOnNameChange}
        onRegexChange={mockOnRegexChange}
        onActionChange={mockOnActionChange}
        onAdd={mockOnAdd}
        onCancel={mockOnCancel}
      />
    );

    // Wait for modal to be visible
    await waitFor(() => {
      expect(screen.getByText("Add custom regex pattern")).toBeInTheDocument();
    });

    // Find and fill the pattern name input
    const nameInput = screen.getByPlaceholderText("e.g., internal_id, employee_code");
    await user.type(nameInput, "employee_id");

    // Find and fill the regex pattern input - use paste instead of type to avoid special char issues
    const regexInput = screen.getByPlaceholderText("e.g., ID-[0-9]{6}");
    await user.click(regexInput);
    await user.paste("EMP-[0-9]{5}");

    // Verify the change handlers were called
    expect(mockOnNameChange).toHaveBeenCalled();
    expect(mockOnRegexChange).toHaveBeenCalled();

    // Find and click the Add button
    const addButton = screen.getByRole("button", { name: /add/i });
    await user.click(addButton);

    // Verify onAdd was called
    expect(mockOnAdd).toHaveBeenCalledTimes(1);
  });
});

