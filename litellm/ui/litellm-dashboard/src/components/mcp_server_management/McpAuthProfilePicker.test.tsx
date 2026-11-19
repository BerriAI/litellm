import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { MCPServer } from "@/components/mcp_tools/types";
import McpAuthProfilePicker from "./McpAuthProfilePicker";

const servers = [
  { server_id: "tickets", server_name: "Tickets", additional_auth: [{ id: "tickets-bots", auth_type: "api_key" }] },
  { server_id: "wiki", server_name: "Wiki", additional_auth: [{ id: "wiki-bots", auth_type: "api_key" }] },
  { server_id: "plain", server_name: "Plain", additional_auth: null },
] as MCPServer[];
describe("McpAuthProfilePicker", () => {
  it("lists every server with profiles, so a grant on a server reached through a team or toolset is visible", () => {
    render(<McpAuthProfilePicker value={["tickets-bots"]} onChange={vi.fn()} servers={servers} />);

    expect(screen.getByLabelText("Tickets")).toHaveTextContent("tickets-bots");
    expect(screen.getByLabelText("Wiki")).toBeInTheDocument();
    expect(screen.queryByLabelText("Plain")).not.toBeInTheDocument();
  });

  it("renders nothing when no server has profiles", () => {
    render(<McpAuthProfilePicker value={[]} onChange={vi.fn()} servers={[servers[2]]} />);

    expect(screen.queryByText("Auth profile per server")).not.toBeInTheDocument();
  });

  it("tells a key with a team that no own grant means the team's profile or the server default", async () => {
    const onChange = vi.fn();
    render(<McpAuthProfilePicker value={["wiki-bots"]} onChange={onChange} servers={servers} followsTeam />);

    expect(screen.getByLabelText("Tickets")).toHaveTextContent("None (team or default)");

    await userEvent.click(screen.getByLabelText("Wiki"));
    await userEvent.click(await screen.findByRole("option", { name: "None (team or default)" }));

    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("shows the granted profile and writes the chosen one, replacing the old", async () => {
    const onChange = vi.fn();
    render(<McpAuthProfilePicker value={["wiki-bots"]} onChange={onChange} servers={servers} />);

    expect(screen.getByLabelText("Wiki")).toHaveTextContent("wiki-bots");
    expect(screen.getByLabelText("Tickets")).toHaveTextContent("Default profile");

    await userEvent.click(screen.getByLabelText("Tickets"));
    await userEvent.click(await screen.findByRole("option", { name: "tickets-bots" }));

    expect(onChange).toHaveBeenCalledWith(["wiki-bots", "tickets-bots"]);
  });
});
