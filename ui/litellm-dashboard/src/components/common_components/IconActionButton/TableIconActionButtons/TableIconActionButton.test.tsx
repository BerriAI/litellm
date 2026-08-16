import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TableIconActionButton, { TableIconActionButtonMap } from "./TableIconActionButton";

describe("TableIconActionButton", () => {
  Object.keys(TableIconActionButtonMap).forEach((variant) => {
    it(`should render ${variant} button`, () => {
      render(<TableIconActionButton variant={variant} onClick={() => {}} dataTestId="test-button" />);
      expect(screen.getByTestId("test-button")).toBeInTheDocument();

      expect(screen.getByTestId("test-button")).toHaveClass(TableIconActionButtonMap[variant].className!);
    });
  });

  it("should have a tooltip", () => {
    render(<TableIconActionButton variant="Edit" onClick={() => {}} dataTestId="test-button" tooltipText="Edit" />);
    const button = screen.getByTestId("test-button");
    const tooltipWrapper = button.closest("span");
    expect(tooltipWrapper).toBeInTheDocument();
  });

  it("should show tooltip when tooltipText is provided", async () => {
    render(
      <TableIconActionButton variant="Edit" onClick={() => {}} dataTestId="test-button" tooltipText="Edit item" />,
    );
    const button = screen.getByTestId("test-button");
    const buttonWrapper = button.closest("span");

    act(() => {
      fireEvent.mouseEnter(buttonWrapper!);
    });

    await waitFor(() => {
      expect(screen.getByText("Edit item")).toBeInTheDocument();
    });
  });

  it("should render disabled state with disabled styling", () => {
    render(
      <TableIconActionButton variant="Edit" onClick={() => {}} dataTestId="test-button" disabled tooltipText="Edit" />,
    );
    const button = screen.getByTestId("test-button");
    expect(button).toHaveClass("opacity-50");
    expect(button).toHaveClass("cursor-not-allowed");
  });

  it("should show disabledTooltipText when disabled and disabledTooltipText is provided", async () => {
    render(
      <TableIconActionButton
        variant="Edit"
        onClick={() => {}}
        dataTestId="test-button"
        disabled
        tooltipText="Edit"
        disabledTooltipText="Cannot edit"
      />,
    );
    const button = screen.getByTestId("test-button");
    const buttonWrapper = button.closest("span");

    act(() => {
      fireEvent.mouseEnter(buttonWrapper!);
    });

    await waitFor(() => {
      expect(screen.getByText("Cannot edit")).toBeInTheDocument();
    });
  });
});
