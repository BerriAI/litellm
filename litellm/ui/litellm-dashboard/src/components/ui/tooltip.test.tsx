import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SimpleTooltip } from "./tooltip";

describe("SimpleTooltip", () => {
  it("should render", () => {
    render(<SimpleTooltip content="Help text" />);
    expect(screen.getByLabelText("question-circle")).toBeInTheDocument();
  });

  it("should render children instead of the default icon when provided", () => {
    render(
      <SimpleTooltip content="Help text">
        <button>Info</button>
      </SimpleTooltip>,
    );
    expect(screen.getByRole("button", { name: /info/i })).toBeInTheDocument();
    expect(screen.queryByLabelText("question-circle")).not.toBeInTheDocument();
  });

  it("should show tooltip content on mouse enter", async () => {
    const user = userEvent.setup();
    render(<SimpleTooltip content="Help text" />);

    await user.hover(screen.getByLabelText("question-circle"));

    expect(screen.getByText("Help text")).toBeInTheDocument();
  });

  it("should hide tooltip content on mouse leave", async () => {
    const user = userEvent.setup();
    render(<SimpleTooltip content="Help text" />);

    await user.hover(screen.getByLabelText("question-circle"));
    expect(screen.getByText("Help text")).toBeInTheDocument();

    await user.unhover(screen.getByLabelText("question-circle"));
    expect(screen.queryByText("Help text")).not.toBeInTheDocument();
  });

  it("should not show tooltip content before hovering", () => {
    render(<SimpleTooltip content="Help text" />);
    expect(screen.queryByText("Help text")).not.toBeInTheDocument();
  });

  it("should keep rendering children when there is no content to show", async () => {
    const user = userEvent.setup();
    render(
      <SimpleTooltip content={undefined}>
        <button>Pick a rubric</button>
      </SimpleTooltip>,
    );

    const trigger = screen.getByRole("button", { name: /pick a rubric/i });
    await user.hover(trigger);

    expect(trigger).toBeInTheDocument();
    expect(document.querySelector('[data-slot="tooltip-content"]')).toBeNull();
  });

  it("should place the tooltip on the requested side", async () => {
    const user = userEvent.setup();
    render(<SimpleTooltip content="Help text" side="right" />);

    await user.hover(screen.getByLabelText("question-circle"));

    expect(await screen.findByText("Help text")).toHaveAttribute("data-side", "right");
  });
});
