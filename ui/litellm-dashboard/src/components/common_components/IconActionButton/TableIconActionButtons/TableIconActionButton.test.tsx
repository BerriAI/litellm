import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import TableIconActionButton, { TableIconActionButtonMap } from "./TableIconActionButton";

describe("TableIconActionButton", () => {
  Object.keys(TableIconActionButtonMap).forEach((variant) => {
    it(`should render ${variant} button`, () => {
      render(<TableIconActionButton variant={variant} onClick={() => {}} dataTestId="test-button" />);
      expect(screen.getByTestId("test-button")).toBeInTheDocument();

      expect(screen.getByTestId("test-button")).toHaveClass(TableIconActionButtonMap[variant].className!);
    });
  });

  it("should call onClick when clicked", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<TableIconActionButton variant="Edit" onClick={onClick} dataTestId="test-button" tooltipText="Edit" />);

    await user.click(screen.getByTestId("test-button"));

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("should not show the tooltip before the button is hovered", () => {
    render(
      <TableIconActionButton variant="Edit" onClick={() => {}} dataTestId="test-button" tooltipText="Edit item" />,
    );
    expect(screen.queryByText("Edit item")).not.toBeInTheDocument();
  });

  it("should show tooltip when tooltipText is provided", async () => {
    const user = userEvent.setup();
    render(
      <TableIconActionButton variant="Edit" onClick={() => {}} dataTestId="test-button" tooltipText="Edit item" />,
    );

    await user.hover(screen.getByTestId("test-button"));

    expect(await screen.findByText("Edit item")).toBeInTheDocument();
  });

  it("should render disabled state with disabled styling", () => {
    render(
      <TableIconActionButton variant="Edit" onClick={() => {}} dataTestId="test-button" disabled tooltipText="Edit" />,
    );
    const button = screen.getByTestId("test-button");
    expect(button).toHaveClass("opacity-50");
    expect(button).toHaveClass("cursor-not-allowed");
  });

  it("should not call onClick when disabled", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <TableIconActionButton
        variant="Edit"
        onClick={onClick}
        dataTestId="test-button"
        disabled
        tooltipText="Edit"
        disabledTooltipText="Cannot edit"
      />,
    );

    await user.click(screen.getByTestId("test-button"));

    expect(onClick).not.toHaveBeenCalled();
  });

  it("should show disabledTooltipText when disabled and disabledTooltipText is provided", async () => {
    const user = userEvent.setup();
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

    await user.hover(screen.getByTestId("test-button"));

    expect(await screen.findByText("Cannot edit")).toBeInTheDocument();
  });
});
