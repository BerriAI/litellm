import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Tooltip } from "./Tooltip";

const getTriggerIcon = (container: HTMLElement) => {
  const svg = container.querySelector("svg.lucide-circle-help, svg.lucide-help-circle");
  if (!svg) throw new Error("Default HelpCircle icon not rendered");
  return svg as SVGElement;
};

describe("Tooltip", () => {
  it("should render", () => {
    const { container } = render(<Tooltip content="Help text" />);
    expect(getTriggerIcon(container)).toBeInTheDocument();
  });

  it("should render children instead of the default icon when provided", () => {
    const { container } = render(
      <Tooltip content="Help text">
        <button>Info</button>
      </Tooltip>,
    );
    expect(screen.getByRole("button", { name: /info/i })).toBeInTheDocument();
    expect(
      container.querySelector("svg.lucide-circle-help, svg.lucide-help-circle"),
    ).not.toBeInTheDocument();
  });

  it("should show tooltip content on mouse enter", async () => {
    const user = userEvent.setup();
    const { container } = render(<Tooltip content="Help text" />);

    await user.hover(getTriggerIcon(container));

    expect(screen.getByText("Help text")).toBeInTheDocument();
  });

  it("should hide tooltip content on mouse leave", async () => {
    const user = userEvent.setup();
    const { container } = render(<Tooltip content="Help text" />);

    await user.hover(getTriggerIcon(container));
    expect(screen.getByText("Help text")).toBeInTheDocument();

    await user.unhover(getTriggerIcon(container));
    expect(screen.queryByText("Help text")).not.toBeInTheDocument();
  });

  it("should not show tooltip content before hovering", () => {
    render(<Tooltip content="Help text" />);
    expect(screen.queryByText("Help text")).not.toBeInTheDocument();
  });
});
