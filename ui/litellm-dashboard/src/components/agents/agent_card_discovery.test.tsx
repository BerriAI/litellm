import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import AgentCardDiscovery from "./agent_card_discovery";

vi.mock("../networking", async () => {
  const actual = await vi.importActual<any>("../networking");
  return {
    ...actual,
    discoverAgentCardCall: vi.fn(),
  };
});

import { discoverAgentCardCall } from "../networking";

const mockDiscover = discoverAgentCardCall as unknown as ReturnType<typeof vi.fn>;

const sampleCard = {
  protocolVersion: "1.0",
  name: "Upstream Agent",
  description: "An upstream agent",
  version: "1.2.3",
  url: "http://internal:9000",
  capabilities: { streaming: true, pushNotifications: true },
  skills: [
    {
      id: "search",
      name: "Search",
      description: "Search the web",
      tags: ["search"],
    },
    {
      id: "summarize",
      name: "Summarize",
      description: "Summarize a document",
      tags: ["llm"],
    },
  ],
  provider: { organization: "UpstreamCo", url: "https://upstream.example" },
};

describe("AgentCardDiscovery", () => {
  beforeEach(() => {
    mockDiscover.mockReset();
  });

  it("renders the URL input and a Discover button", () => {
    renderWithProviders(
      <AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />,
    );

    expect(
      screen.getByPlaceholderText("https://upstream-agent.example.com"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /discover/i })).toBeInTheDocument();
  });

  it("shows an error when discover is clicked without a URL", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />,
    );

    await user.click(screen.getByRole("button", { name: /discover/i }));
    expect(
      await screen.findByText(/Enter the agent's base URL first/i),
    ).toBeInTheDocument();
    expect(mockDiscover).not.toHaveBeenCalled();
  });

  it("renders the upstream skills and capabilities on success", async () => {
    mockDiscover.mockResolvedValueOnce({
      url: "https://upstream.example.com",
      agent_card: sampleCard,
    });
    const user = userEvent.setup();
    renderWithProviders(
      <AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />,
    );

    await user.type(
      screen.getByPlaceholderText("https://upstream-agent.example.com"),
      "https://upstream.example.com",
    );
    await user.click(screen.getByRole("button", { name: /discover/i }));

    expect(await screen.findByText("Upstream card loaded")).toBeInTheDocument();
    expect(screen.getByText("Search")).toBeInTheDocument();
    expect(screen.getByText("Summarize")).toBeInTheDocument();
    // Only proxy-supported capabilities surface (streaming).
    expect(screen.getByText(/^streaming$/i)).toBeInTheDocument();
    expect(screen.queryByText(/pushNotifications/i)).not.toBeInTheDocument();
  });

  it("shows an inline error when discovery fails", async () => {
    mockDiscover.mockRejectedValueOnce(new Error("upstream unreachable"));
    const user = userEvent.setup();
    renderWithProviders(
      <AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />,
    );

    await user.type(
      screen.getByPlaceholderText("https://upstream-agent.example.com"),
      "https://nope.example",
    );
    await user.click(screen.getByRole("button", { name: /discover/i }));

    expect(await screen.findByText("Discovery failed")).toBeInTheDocument();
    expect(screen.getByText(/upstream unreachable/)).toBeInTheDocument();
  });

  it("emits the selected subset when the user applies the card", async () => {
    mockDiscover.mockResolvedValueOnce({
      url: "https://upstream.example.com",
      agent_card: sampleCard,
    });
    const onApply = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <AgentCardDiscovery accessToken="tok" onApply={onApply} />,
    );

    await user.type(
      screen.getByPlaceholderText("https://upstream-agent.example.com"),
      "https://upstream.example.com",
    );
    await user.click(screen.getByRole("button", { name: /discover/i }));
    await screen.findByText("Upstream card loaded");

    // Deselect the "Summarize" skill by clicking its row's checkbox.
    const summarizeLabel = screen.getByText("Summarize").closest("label");
    expect(summarizeLabel).toBeTruthy();
    const summarizeCheckbox = summarizeLabel!.querySelector(
      "input[type='checkbox']",
    ) as HTMLInputElement;
    await user.click(summarizeCheckbox);

    await user.click(screen.getByRole("button", { name: /use these selections/i }));

    await waitFor(() => expect(onApply).toHaveBeenCalledTimes(1));
    const selection = onApply.mock.calls[0][0];
    expect(selection.upstream_url).toBe("https://upstream.example.com");
    expect(selection.raw_card).toEqual(sampleCard);
    expect(selection.selected_card.skills).toHaveLength(1);
    expect(selection.selected_card.skills[0].id).toBe("search");
    expect(selection.selected_card.name).toBe("Upstream Agent");
  });

  it("hides the URL input and shows the display URL when parent-driven", () => {
    renderWithProviders(
      <AgentCardDiscovery
        accessToken="tok"
        onApply={vi.fn()}
        discoveryRequest={{
          url: "http://localhost:2024",
          discovery_mode: "langgraph_platform",
          params: { assistant_id: "agent" },
          display_url:
            "http://localhost:2024/.well-known/agent-card.json?assistant_id=agent",
        }}
      />,
    );

    // Free-form URL input is gone.
    expect(
      screen.queryByPlaceholderText("https://upstream-agent.example.com"),
    ).not.toBeInTheDocument();
    // The exact URL the proxy will hit is visible.
    expect(
      screen.getByText(
        "http://localhost:2024/.well-known/agent-card.json?assistant_id=agent",
      ),
    ).toBeInTheDocument();
  });

  it("forwards discovery_mode and params from the parent plan", async () => {
    mockDiscover.mockResolvedValueOnce({
      url: "http://localhost:2024",
      agent_card: sampleCard,
    });
    const user = userEvent.setup();
    renderWithProviders(
      <AgentCardDiscovery
        accessToken="tok"
        onApply={vi.fn()}
        discoveryRequest={{
          url: "http://localhost:2024",
          discovery_mode: "langgraph_platform",
          params: { assistant_id: "agent" },
          display_url:
            "http://localhost:2024/.well-known/agent-card.json?assistant_id=agent",
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: /discover/i }));

    await waitFor(() => expect(mockDiscover).toHaveBeenCalledTimes(1));
    expect(mockDiscover).toHaveBeenCalledWith("tok", "http://localhost:2024", {
      discovery_mode: "langgraph_platform",
      params: { assistant_id: "agent" },
    });
  });

  it("disables Discover until the parent provides a usable URL", async () => {
    renderWithProviders(
      <AgentCardDiscovery
        accessToken="tok"
        onApply={vi.fn()}
        discoveryRequest={{
          url: "",
          discovery_mode: "langgraph_platform",
          params: { assistant_id: "" },
          display_url: "",
        }}
      />,
    );

    expect(
      (screen.getByRole("button", {
        name: /discover/i,
      }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("blocks discover when no access token is provided", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <AgentCardDiscovery accessToken={null} onApply={vi.fn()} />,
    );

    await user.type(
      screen.getByPlaceholderText("https://upstream-agent.example.com"),
      "https://upstream.example.com",
    );
    await user.click(screen.getByRole("button", { name: /discover/i }));

    expect(
      await screen.findByText(/No access token available/i),
    ).toBeInTheDocument();
    expect(mockDiscover).not.toHaveBeenCalled();
  });
});
