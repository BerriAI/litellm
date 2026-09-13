import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import MCPServerCostDisplay from "./mcp_server_cost_display";

describe("MCPServerCostDisplay", () => {
  it("explains that calls are free when no cost config exists", () => {
    render(<MCPServerCostDisplay costConfig={null} />);

    expect(
      screen.getByText("No cost configuration set for this server. Tool calls will be charged at $0.00 per tool call."),
    ).toBeInTheDocument();
  });

  it("treats a config with only a null default cost as unconfigured", () => {
    render(<MCPServerCostDisplay costConfig={{ default_cost_per_query: null }} />);

    expect(screen.getByText(/No cost configuration set for this server/)).toBeInTheDocument();
  });

  it("shows a zero default cost rather than falling back to the empty state", () => {
    render(<MCPServerCostDisplay costConfig={{ default_cost_per_query: 0 }} />);

    expect(screen.getByText("Default Cost per Query")).toBeInTheDocument();
    expect(screen.getByText("$0.0000")).toBeInTheDocument();
  });

  it("renders the default cost to four decimal places and summarises it", () => {
    render(<MCPServerCostDisplay costConfig={{ default_cost_per_query: 0.0125 }} />);

    expect(screen.getByText("$0.0125")).toBeInTheDocument();
    expect(screen.getByText("• Default cost: $0.0125 per query")).toBeInTheDocument();
  });

  it("lists each tool-specific cost and counts them in the summary", () => {
    render(
      <MCPServerCostDisplay
        costConfig={{ tool_name_to_cost_per_query: { search: 0.5, fetch: 0.25, skipped: null } }}
      />,
    );

    expect(screen.getByText("search")).toBeInTheDocument();
    expect(screen.getByText("$0.5000 per query")).toBeInTheDocument();
    expect(screen.getByText("fetch")).toBeInTheDocument();
    expect(screen.getByText("$0.2500 per query")).toBeInTheDocument();
    expect(screen.queryByText("skipped")).not.toBeInTheDocument();
    expect(screen.getByText("• 3 tool(s) with custom pricing")).toBeInTheDocument();
  });
});
