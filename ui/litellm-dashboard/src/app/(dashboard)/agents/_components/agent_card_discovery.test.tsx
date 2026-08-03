import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import AgentCardDiscovery from "./agent_card_discovery";

vi.mock("@/components/networking", async () => {
  const actual = await vi.importActual<any>("@/components/networking");
  return {
    ...actual,
    discoverAgentCardCall: vi.fn(),
  };
});

import { discoverAgentCardCall } from "@/components/networking";

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
    vi.useFakeTimers({ shouldAdvanceTime: true });
    mockDiscover.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the URL input and a Re-discover button after manual entry", async () => {
    mockDiscover.mockResolvedValue({
      url: "https://upstream.example.com",
      agent_card: sampleCard,
    });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderWithProviders(<AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />);

    expect(screen.getByPlaceholderText("https://upstream-agent.example.com")).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText("https://upstream-agent.example.com"), "https://upstream.example.com");
    await vi.advanceTimersByTimeAsync(500);

    await waitFor(() => expect(mockDiscover).toHaveBeenCalled());
    expect(await screen.findByRole("button", { name: /re-discover/i })).toBeInTheDocument();
  });

  it("shows an error when re-discover is clicked without a URL", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderWithProviders(<AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: /discover/i }));
    expect(await screen.findByText(/Enter the agent's base URL first/i)).toBeInTheDocument();
    expect(mockDiscover).not.toHaveBeenCalled();
  });

  it("auto-discovers and renders upstream skills on success", async () => {
    mockDiscover.mockResolvedValueOnce({
      url: "https://upstream.example.com",
      agent_card: sampleCard,
    });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderWithProviders(<AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />);

    await user.type(screen.getByPlaceholderText("https://upstream-agent.example.com"), "https://upstream.example.com");
    await vi.advanceTimersByTimeAsync(500);

    expect(await screen.findByText("Upstream card loaded")).toBeInTheDocument();
    expect(screen.getByText("Search")).toBeInTheDocument();
    expect(screen.getByText("Summarize")).toBeInTheDocument();
    expect(screen.getByText(/^streaming$/i)).toBeInTheDocument();
    expect(screen.queryByText(/pushNotifications/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /use these selections/i })).not.toBeInTheDocument();
  });

  it("shows an inline error when discovery fails", async () => {
    mockDiscover.mockRejectedValueOnce(new Error("upstream unreachable"));
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderWithProviders(<AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />);

    await user.type(screen.getByPlaceholderText("https://upstream-agent.example.com"), "https://nope.example");
    await vi.advanceTimersByTimeAsync(500);

    expect(await screen.findByText("Discovery failed")).toBeInTheDocument();
    expect(screen.getByText(/upstream unreachable/)).toBeInTheDocument();
  });

  it("syncs the selected subset to the parent as the user edits", async () => {
    mockDiscover.mockResolvedValueOnce({
      url: "https://upstream.example.com",
      agent_card: sampleCard,
    });
    const onApply = vi.fn();
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderWithProviders(<AgentCardDiscovery accessToken="tok" onApply={onApply} />);

    await user.type(screen.getByPlaceholderText("https://upstream-agent.example.com"), "https://upstream.example.com");
    await vi.advanceTimersByTimeAsync(500);
    await screen.findByText("Upstream card loaded");

    await waitFor(() => expect(onApply).toHaveBeenCalled());
    const initialSelection = onApply.mock.calls.at(-1)?.[0];
    expect(initialSelection.upstream_url).toBe("https://upstream.example.com");
    expect(initialSelection.selected_card.skills).toHaveLength(2);

    await user.click(screen.getByRole("checkbox", { name: /Summarize/i }));

    await waitFor(() => {
      const latest = onApply.mock.calls.at(-1)?.[0];
      expect(latest.selected_card.skills).toHaveLength(1);
      expect(latest.selected_card.skills[0].id).toBe("search");
    });
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
          display_url: "http://localhost:2024/.well-known/agent-card.json?assistant_id=agent",
        }}
      />,
    );

    expect(screen.queryByPlaceholderText("https://upstream-agent.example.com")).not.toBeInTheDocument();
    expect(
      screen.getByText("http://localhost:2024/.well-known/agent-card.json?assistant_id=agent"),
    ).toBeInTheDocument();
  });

  it("auto-discovers with discovery_mode and params from the parent plan", async () => {
    mockDiscover.mockResolvedValueOnce({
      url: "http://localhost:2024",
      agent_card: sampleCard,
    });
    renderWithProviders(
      <AgentCardDiscovery
        accessToken="tok"
        onApply={vi.fn()}
        discoveryRequest={{
          url: "http://localhost:2024",
          discovery_mode: "langgraph_platform",
          params: { assistant_id: "agent" },
          display_url: "http://localhost:2024/.well-known/agent-card.json?assistant_id=agent",
        }}
      />,
    );

    await vi.advanceTimersByTimeAsync(0);
    await waitFor(() => expect(mockDiscover).toHaveBeenCalledTimes(1));
    expect(mockDiscover).toHaveBeenCalledWith("tok", "http://localhost:2024", {
      discovery_mode: "langgraph_platform",
      params: { assistant_id: "agent" },
    });
  });

  it("disables Re-discover until the parent provides a usable URL", async () => {
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
      (
        screen.getByRole("button", {
          name: /discover/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(mockDiscover).not.toHaveBeenCalled();
  });

  it("pre-selects only skills present in savedAgentCard when editing", async () => {
    mockDiscover.mockResolvedValueOnce({
      url: "http://localhost:2024",
      agent_card: sampleCard,
    });
    const onApply = vi.fn();
    renderWithProviders(
      <AgentCardDiscovery
        accessToken="tok"
        onApply={onApply}
        discoveryRequest={{
          url: "http://localhost:2024",
          discovery_mode: "langgraph_platform",
          params: { assistant_id: "agent" },
        }}
        savedAgentCard={{
          name: "DB Agent",
          description: "DB description",
          capabilities: { streaming: false },
          skills: [{ id: "search", name: "Search" }],
        }}
      />,
    );

    await vi.advanceTimersByTimeAsync(0);
    await screen.findByText("Upstream card loaded");

    await waitFor(() => expect(onApply).toHaveBeenCalled());
    const selection = onApply.mock.calls.at(-1)?.[0];
    expect(selection.selected_card.skills).toHaveLength(1);
    expect(selection.selected_card.skills[0].id).toBe("search");
    expect(selection.selected_card.name).toBe("DB Agent");
    expect(selection.selected_card.capabilities.streaming).toBe(false);
  });

  it("does not fire discovery before the debounce wait and fires once with the last URL", async () => {
    mockDiscover.mockResolvedValue({
      url: "https://last.example.com",
      agent_card: sampleCard,
    });
    renderWithProviders(<AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />);
    const input = screen.getByPlaceholderText("https://upstream-agent.example.com");

    act(() => {
      fireEvent.change(input, { target: { value: "https://first.example.com" } });
    });
    act(() => {
      vi.advanceTimersByTime(399);
    });
    expect(mockDiscover).not.toHaveBeenCalled();

    act(() => {
      fireEvent.change(input, { target: { value: "https://last.example.com" } });
    });
    act(() => {
      vi.advanceTimersByTime(399);
    });
    expect(mockDiscover).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(mockDiscover).toHaveBeenCalledTimes(1);
    expect(mockDiscover).toHaveBeenCalledWith("tok", "https://last.example.com", undefined);
  });

  it("fires no discovery when unmounted mid-wait", () => {
    const { unmount } = renderWithProviders(<AgentCardDiscovery accessToken="tok" onApply={vi.fn()} />);
    const input = screen.getByPlaceholderText("https://upstream-agent.example.com");

    act(() => {
      fireEvent.change(input, { target: { value: "https://first.example.com" } });
    });
    act(() => {
      vi.advanceTimersByTime(200);
    });
    unmount();
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(mockDiscover).not.toHaveBeenCalled();
  });

  it("blocks discover when no access token is provided", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderWithProviders(<AgentCardDiscovery accessToken={null} onApply={vi.fn()} />);

    await user.type(screen.getByPlaceholderText("https://upstream-agent.example.com"), "https://upstream.example.com");
    await user.click(screen.getByRole("button", { name: /discover/i }));

    expect(await screen.findByText(/No access token available/i)).toBeInTheDocument();
    expect(mockDiscover).not.toHaveBeenCalled();
  });
});
