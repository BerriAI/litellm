import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { MessageCard, ToolResultCard } from "./Messages";
import { Section } from "./Section";

vi.mock("../../ui/spanProvider", () => ({ useSpanProvider: () => null }));

const LONG_QUERY = "Find every invoice for the customer that was billed twice. ".repeat(3).trim();

describe("ToolResultCard", () => {
  it("shows multiline output immediately and expands long results without losing text", async () => {
    const user = userEvent.setup();
    const result = Array.from({ length: 30 }, (_, i) => `line ${i + 1}`).join("\n");
    render(
      <ToolResultCard name="terminal" result={JSON.stringify({ output: result, exit_code: 1, error: null })} failed />,
    );
    expect(screen.getByText(/line 1\s+line 2/, { selector: "pre" })).toBeVisible();
    expect(screen.queryByText(/line 30/, { selector: "pre" })).not.toBeInTheDocument();
    const failedResult = screen.getByRole("group", { name: "Failed tool result" });
    expect(failedResult).toHaveClass("text-destructive");
    expect(within(failedResult).getByText("exit_code")).toBeVisible();
    expect(within(failedResult).getByText("1", { exact: true })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Expand result" }));
    expect(screen.getByText(/line 30/, { selector: "pre" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Collapse result" }));
    expect(screen.queryByText(/line 30/, { selector: "pre" })).not.toBeInTheDocument();
  });
});

describe("MessageCard", () => {
  it("folds the card body from the role tile and restores it", async () => {
    const user = userEvent.setup();
    render(<MessageCard message={{ role: "user", content: "Why was I billed twice?" }} />);
    expect(screen.getByText("Why was I billed twice?")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Collapse User" }));
    expect(screen.queryByText("Why was I billed twice?")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Expand User" }));
    expect(screen.getByText("Why was I billed twice?")).toBeVisible();
  });

  it("shows a tool's primary query immediately alongside its other arguments", () => {
    render(
      <MessageCard
        message={{
          role: "assistant",
          content: "",
          tool_calls: [{ name: "search_invoices", args: { customer_id: "test-404", query: LONG_QUERY } }],
        }}
      />,
    );
    expect(screen.getByText("test-404")).toBeVisible();
    expect(screen.getByText(LONG_QUERY, { selector: "pre" })).toBeVisible();
  });

  it("shows a tool result as one line with the tool name and its output", () => {
    render(<MessageCard message={{ role: "tool", name: "ls", content: "No files found" }} />);
    expect(screen.getByText("ls")).toBeInTheDocument();
    expect(screen.getByText("No files found")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy ls result" })).toBeInTheDocument();
  });
});

describe("Section", () => {
  it("hides its body immediately when collapsed", async () => {
    const user = userEvent.setup();
    render(
      <Section title="Output">
        <p>final answer</p>
      </Section>,
    );
    await user.click(screen.getByRole("button", { name: "Output" }));
    expect(screen.getByRole("button", { name: "Output" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("final answer")).not.toBeVisible();
  });
});
