import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import MCPDiscovery from "./mcp_discovery";
import { fetchDiscoverableMCPServers } from "@/components/networking";
import type { DiscoverableMCPServer } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({
  fetchDiscoverableMCPServers: vi.fn(),
}));

const githubServer = {
  name: "github",
  title: "GitHub",
  description: "Code hosting",
  category: "Developer Tools",
  icon_url: "",
} as DiscoverableMCPServer;

const slackServer = {
  name: "slack",
  title: "Slack",
  description: "Team chat",
  category: "Communication",
  icon_url: "",
} as DiscoverableMCPServer;

const defaultProps = {
  isVisible: true,
  onClose: vi.fn(),
  onSelectServer: vi.fn(),
  onCustomServer: vi.fn(),
  accessToken: "tok",
};

describe("MCPDiscovery", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchDiscoverableMCPServers).mockResolvedValue({
      servers: [githubServer, slackServer],
      categories: ["Developer Tools", "Communication"],
    });
  });

  // Each category name renders twice: once as a filter pill (a button) and once
  // as the heading of its group. Only the heading is not a button.
  const groupHeading = (category: string) => screen.getAllByText(category).filter((el) => el.tagName !== "BUTTON");

  it("lists every discoverable server grouped under its category", async () => {
    render(<MCPDiscovery {...defaultProps} />);

    expect(await screen.findByText("GitHub")).toBeInTheDocument();
    expect(screen.getByText("Slack")).toBeInTheDocument();
    expect(groupHeading("Developer Tools")).toHaveLength(1);
    expect(groupHeading("Communication")).toHaveLength(1);
    expect(screen.getByText("Add MCP Server")).toBeInTheDocument();
  });

  it("filters the list down to the chosen category", async () => {
    render(<MCPDiscovery {...defaultProps} />);
    await screen.findByText("GitHub");

    await userEvent.click(screen.getByRole("button", { name: "Communication" }));

    await waitFor(() => expect(screen.queryByText("GitHub")).not.toBeInTheDocument());
    expect(screen.getByText("Slack")).toBeInTheDocument();
  });

  it("filters the list by the search term", async () => {
    render(<MCPDiscovery {...defaultProps} />);
    await screen.findByText("GitHub");

    await userEvent.type(screen.getByPlaceholderText("Search servers..."), "chat");

    await waitFor(() => expect(screen.queryByText("GitHub")).not.toBeInTheDocument());
    expect(screen.getByText("Slack")).toBeInTheDocument();
  });

  it("hands the picked server back to the caller", async () => {
    const onSelectServer = vi.fn();
    render(<MCPDiscovery {...defaultProps} onSelectServer={onSelectServer} />);

    await userEvent.click(await screen.findByText("GitHub"));

    expect(onSelectServer).toHaveBeenCalledWith(githubServer);
  });

  it("offers a custom-server escape hatch", async () => {
    const onCustomServer = vi.fn();
    render(<MCPDiscovery {...defaultProps} onCustomServer={onCustomServer} />);

    await userEvent.click(await screen.findByRole("button", { name: "+ Custom Server" }));

    expect(onCustomServer).toHaveBeenCalled();
  });

  it("surfaces a fetch failure", async () => {
    vi.mocked(fetchDiscoverableMCPServers).mockRejectedValue(new Error("registry down"));

    render(<MCPDiscovery {...defaultProps} />);

    expect(await screen.findByText(/Failed to load servers: registry down/)).toBeInTheDocument();
  });

  it("offers the custom-server link when nothing matches", async () => {
    vi.mocked(fetchDiscoverableMCPServers).mockResolvedValue({ servers: [], categories: [] });

    render(<MCPDiscovery {...defaultProps} />);

    expect(await screen.findByText(/No servers found/)).toBeInTheDocument();
  });

  it("does not fetch while hidden", () => {
    render(<MCPDiscovery {...defaultProps} isVisible={false} />);

    expect(fetchDiscoverableMCPServers).not.toHaveBeenCalled();
  });
});
