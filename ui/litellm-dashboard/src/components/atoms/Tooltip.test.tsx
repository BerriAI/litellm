import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Tooltip } from "./Tooltip";

vi.mock("@ant-design/icons", () => ({
  QuestionCircleOutlined: (props: any) => <span data-testid="question-icon" {...props} />,
}));

describe("Tooltip", () => {
  it("should render", () => {
    render(<Tooltip content="Help text" />);
    expect(screen.getByTestId("question-icon")).toBeInTheDocument();
  });

  it("should render children instead of the default icon when provided", () => {
    render(<Tooltip content="Help text"><button>Info</button></Tooltip>);
    expect(screen.getByRole("button", { name: /info/i })).toBeInTheDocument();
    expect(screen.queryByTestId("question-icon")).not.toBeInTheDocument();
  });

  it("should show tooltip content on mouse enter", async () => {
    const user = userEvent.setup();
    render(<Tooltip content="Help text" />);

    await user.hover(screen.getByTestId("question-icon"));

    expect(screen.getByText("Help text")).toBeInTheDocument();
  });

  it("should hide tooltip content on mouse leave", async () => {
    const user = userEvent.setup();
    render(<Tooltip content="Help text" />);

    await user.hover(screen.getByTestId("question-icon"));
    expect(screen.getByText("Help text")).toBeInTheDocument();

    await user.unhover(screen.getByTestId("question-icon"));
    expect(screen.queryByText("Help text")).not.toBeInTheDocument();
  });

  it("should not show tooltip content before hovering", () => {
    render(<Tooltip content="Help text" />);
    expect(screen.queryByText("Help text")).not.toBeInTheDocument();
  });
});
