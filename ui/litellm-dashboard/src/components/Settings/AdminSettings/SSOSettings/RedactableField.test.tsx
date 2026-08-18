import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";
import RedactableField from "./RedactableField";

describe("RedactableField", () => {
  describe("when value is null", () => {
    it("should display 'Not configured' text", () => {
      renderWithProviders(<RedactableField value={null} />);

      expect(screen.getByText("Not configured")).toBeInTheDocument();
    });

    it("should not display toggle button", () => {
      renderWithProviders(<RedactableField value={null} />);
      const buttons = screen.queryAllByRole("button");
      expect(buttons).toHaveLength(0);
    });
  });

  describe("when value is provided", () => {
    const testValue = "secret-password";

    it("should be hidden by default and show redacted dots", () => {
      renderWithProviders(<RedactableField value={testValue} />);
      expect(screen.getByText("•".repeat(testValue.length))).toBeInTheDocument();
      expect(screen.queryByText(testValue)).not.toBeInTheDocument();
    });

    it("should show actual value when defaultHidden is false", () => {
      renderWithProviders(<RedactableField value={testValue} defaultHidden={false} />);

      expect(screen.getByText(testValue)).toBeInTheDocument();
      expect(screen.queryByText("•".repeat(testValue.length))).not.toBeInTheDocument();
    });

    it("should identify the hidden-value control and render its icon", () => {
      renderWithProviders(<RedactableField value={testValue} />);

      const button = screen.getByRole("button", { name: "Show value" });
      expect(button.querySelector("svg")).toBeInTheDocument();
    });

    it("should identify the visible-value control and render its icon", () => {
      renderWithProviders(<RedactableField value={testValue} defaultHidden={false} />);

      const button = screen.getByRole("button", { name: "Hide value" });
      expect(button.querySelector("svg")).toBeInTheDocument();
    });

    it("should toggle visibility when button is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<RedactableField value={testValue} />);

      await user.click(screen.getByRole("button", { name: "Show value" }));
      expect(screen.getByText(testValue)).toBeInTheDocument();
      expect(screen.queryByText("•".repeat(testValue.length))).not.toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Hide value" }));
      expect(screen.getByText("•".repeat(testValue.length))).toBeInTheDocument();
      expect(screen.queryByText(testValue)).not.toBeInTheDocument();
    });

    it("should handle empty string value", () => {
      renderWithProviders(<RedactableField value="" />);
      expect(screen.getByText("Not configured")).toBeInTheDocument();

      const buttons = screen.queryAllByRole("button");
      expect(buttons).toHaveLength(0);
    });

    it("should handle different value lengths correctly", () => {
      const shortValue = "hi";
      const longValue = "this-is-a-very-long-secret-value";

      const { rerender } = renderWithProviders(<RedactableField value={shortValue} />);
      expect(screen.getByText("••")).toBeInTheDocument();

      rerender(<RedactableField value={longValue} />);
      expect(screen.getByText("•".repeat(longValue.length))).toBeInTheDocument();
    });
  });
});
