import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SectionHeader } from "./SectionHeader";

describe("SectionHeader", () => {
  it("renders the input label with token, cost and turn metrics", () => {
    render(<SectionHeader type="input" tokens={1234} cost={0.000123} turnCount={3} onCopy={vi.fn()} />);

    expect(screen.getByText("Input")).toBeInTheDocument();
    expect(screen.getByText("Tokens: 1,234")).toBeInTheDocument();
    expect(screen.getByText("Cost: $0.000123")).toBeInTheDocument();
    expect(screen.getByText("Turns: 3")).toBeInTheDocument();
  });

  it("renders the output label", () => {
    render(<SectionHeader type="output" onCopy={vi.fn()} />);

    expect(screen.getByText("Output")).toBeInTheDocument();
  });

  it("omits metrics that were not provided", () => {
    render(<SectionHeader type="input" onCopy={vi.fn()} />);

    expect(screen.queryByText(/^Tokens:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Cost:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Turns:/)).not.toBeInTheDocument();
  });

  it("omits the turn count when there are no turns", () => {
    render(<SectionHeader type="input" turnCount={0} onCopy={vi.fn()} />);

    expect(screen.queryByText(/^Turns:/)).not.toBeInTheDocument();
  });

  it("copies without toggling the section", async () => {
    const onCopy = vi.fn();
    const onToggleCollapse = vi.fn();
    render(<SectionHeader type="input" onCopy={onCopy} onToggleCollapse={onToggleCollapse} />);

    await userEvent.click(screen.getByRole("button", { name: /copy/i }));

    expect(onCopy).toHaveBeenCalledTimes(1);
    expect(onToggleCollapse).not.toHaveBeenCalled();
  });

  it("toggles the section when the header is clicked", async () => {
    const onToggleCollapse = vi.fn();
    render(<SectionHeader type="input" onCopy={vi.fn()} onToggleCollapse={onToggleCollapse} />);

    await userEvent.click(screen.getByText("Input"));

    expect(onToggleCollapse).toHaveBeenCalledTimes(1);
  });

  it("stays inert when no toggle handler is given", async () => {
    const onCopy = vi.fn();
    render(<SectionHeader type="input" onCopy={onCopy} />);

    await userEvent.click(screen.getByText("Input"));

    expect(onCopy).not.toHaveBeenCalled();
  });
});
