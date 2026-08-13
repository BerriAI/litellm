import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RedactableField from "./RedactableField";

describe("RedactableField", () => {
  describe("when value is null", () => {
    it("should display 'Not configured' text", () => {
      render(<RedactableField value={null} />);

      expect(screen.getByText("Not configured")).toBeInTheDocument();
    });

    it("should not display toggle button", () => {
      render(<RedactableField value={null} />);

      // There should be no button elements
      const buttons = screen.queryAllByRole("button");
      expect(buttons).toHaveLength(0);
    });
  });

  describe("when value is provided", () => {
    const testValue = "secret-password";

    it("should be hidden by default and show redacted dots", () => {
      render(<RedactableField value={testValue} />);

      // Should show dots equal to the length of the value
      expect(screen.getByText("•".repeat(testValue.length))).toBeInTheDocument();
      expect(screen.queryByText(testValue)).not.toBeInTheDocument();
    });

    it("should show actual value when defaultHidden is false", () => {
      render(<RedactableField value={testValue} defaultHidden={false} />);

      expect(screen.getByText(testValue)).toBeInTheDocument();
      expect(screen.queryByText("•".repeat(testValue.length))).not.toBeInTheDocument();
    });

    it("should display toggle button with eye icon when hidden", () => {
      render(<RedactableField value={testValue} />);

      const button = screen.getByRole("button");
      expect(button).toBeInTheDocument();

      // Check that the Eye icon is rendered (we can check by title or by the presence of the icon)
      // The button should contain the Eye icon when hidden
      const eyeIcon = button.querySelector("svg");
      expect(eyeIcon).toBeInTheDocument();
    });

    it("should display toggle button with eye-off icon when shown", () => {
      render(<RedactableField value={testValue} defaultHidden={false} />);

      const button = screen.getByRole("button");
      expect(button).toBeInTheDocument();

      // The button should contain the EyeOff icon when shown
      const eyeOffIcon = button.querySelector("svg");
      expect(eyeOffIcon).toBeInTheDocument();
    });

    it("should toggle visibility when button is clicked", () => {
      render(<RedactableField value={testValue} />);

      // Initially hidden
      expect(screen.getByText("•".repeat(testValue.length))).toBeInTheDocument();
      expect(screen.queryByText(testValue)).not.toBeInTheDocument();

      // Click to show
      const button = screen.getByRole("button");
      fireEvent.click(button);

      // Should now show the actual value
      expect(screen.getByText(testValue)).toBeInTheDocument();
      expect(screen.queryByText("•".repeat(testValue.length))).not.toBeInTheDocument();

      // Click again to hide
      fireEvent.click(button);

      // Should be hidden again
      expect(screen.getByText("•".repeat(testValue.length))).toBeInTheDocument();
      expect(screen.queryByText(testValue)).not.toBeInTheDocument();
    });

    it("should handle empty string value", () => {
      render(<RedactableField value="" />);

      // Empty string should show "Not configured" since value is falsy
      expect(screen.getByText("Not configured")).toBeInTheDocument();

      // No toggle button for empty string
      const buttons = screen.queryAllByRole("button");
      expect(buttons).toHaveLength(0);
    });

    it("should handle different value lengths correctly", () => {
      const shortValue = "hi";
      const longValue = "this-is-a-very-long-secret-value";

      const { rerender } = render(<RedactableField value={shortValue} />);
      expect(screen.getByText("••")).toBeInTheDocument();

      rerender(<RedactableField value={longValue} />);
      expect(screen.getByText("•".repeat(longValue.length))).toBeInTheDocument();
    });
  });
});
