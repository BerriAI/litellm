import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { MCPEvent } from "@/components/mcp_tools/types";
import MCPEventsDisplay from "./MCPEventsDisplay";

const toolsEvent: MCPEvent = {
  type: "response.output_item.done",
  item: {
    type: "mcp_list_tools",
    tools: [{ name: "available_lookup", description: "Look up a document" }],
  },
};

const callEvent: MCPEvent = {
  type: "response.output_item.done",
  item: {
    type: "mcp_call",
    name: "document_lookup",
    arguments: '{"query":"chat"}',
    output: "Found the requested document",
  },
};

describe("MCPEventsDisplay", () => {
  it("collapses discovery while showing tool calls and lets users expand discovery", async () => {
    render(<MCPEventsDisplay events={[toolsEvent, callEvent]} />);

    const listTools = screen.getByRole("button", { name: "List tools" });
    expect(listTools).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("available_lookup")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "document_lookup" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Found the requested document")).toBeVisible();

    fireEvent.click(listTools);
    expect(await screen.findByText("available_lookup")).toBeVisible();
    expect(listTools).toHaveAttribute("aria-expanded", "true");
  });

  it("keeps a tool call expanded when no discovery event is present", () => {
    render(<MCPEventsDisplay events={[callEvent]} />);
    expect(screen.getByRole("button", { name: "document_lookup" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Found the requested document")).toBeVisible();
  });

  it("opens streamed tool calls without overriding the user's collapsed panels", () => {
    const { rerender } = render(<MCPEventsDisplay events={[toolsEvent]} />);
    rerender(<MCPEventsDisplay events={[toolsEvent, callEvent]} />);

    const toolCall = screen.getByRole("button", { name: "document_lookup" });
    expect(toolCall).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(toolCall);

    const nextCall: MCPEvent = {
      ...callEvent,
      item: { ...callEvent.item, name: "next_lookup" },
    };
    rerender(<MCPEventsDisplay events={[toolsEvent, callEvent, nextCall]} />);

    expect(toolCall).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: "List tools" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: "next_lookup" })).toHaveAttribute("aria-expanded", "true");
  });
});
