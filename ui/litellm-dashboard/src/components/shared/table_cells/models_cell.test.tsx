import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ModelsCell } from "./models_cell";

describe("ModelsCell", () => {
  it("shows 'All Proxy Models' when the list is empty, null, or undefined", () => {
    const { rerender } = render(<ModelsCell models={[]} />);
    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
    rerender(<ModelsCell models={null} />);
    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
    rerender(<ModelsCell models={undefined} />);
    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
  });

  it("renders every model with no overflow badge when at or below the limit", () => {
    render(<ModelsCell models={["gpt-4o", "claude-sonnet-4-5", "o3-mini"]} maxVisible={3} />);
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet-4-5")).toBeInTheDocument();
    expect(screen.getByText("o3-mini")).toBeInTheDocument();
    expect(screen.queryByText(/more$/)).not.toBeInTheDocument();
  });

  it("collapses models beyond the limit into a '+N more' badge", () => {
    render(<ModelsCell models={["a", "b", "c", "d", "e"]} maxVisible={2} />);
    expect(screen.getByText("a")).toBeInTheDocument();
    expect(screen.getByText("b")).toBeInTheDocument();
    expect(screen.queryByText("c")).not.toBeInTheDocument();
    expect(screen.getByText("+3 more")).toBeInTheDocument();
  });

  it("reveals the hidden models in a tooltip on hover", async () => {
    const user = userEvent.setup();
    render(<ModelsCell models={["a", "b", "c", "d"]} maxVisible={2} />);
    await user.hover(screen.getByText("+2 more"));
    expect(await screen.findByText("c")).toBeInTheDocument();
    expect(await screen.findByText("d")).toBeInTheDocument();
  });

  it("labels the all-proxy-models wildcard", () => {
    render(<ModelsCell models={["all-proxy-models"]} />);
    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
  });
});
