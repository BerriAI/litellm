import React from "react";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AgentsPanel from "./AgentsPanel";
import * as networking from "@/components/networking";
import { renderWithProviders } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  getAgentsList: vi.fn().mockResolvedValue({ agents: [] }),
  deleteAgentCall: vi.fn().mockResolvedValue({}),
}));

vi.mock("./add_agent_form", () => ({
  default: ({ onSuccess }: { onSuccess: () => void }) => (
    <button data-testid="add-agent-form" onClick={onSuccess}>
      Finish add
    </button>
  ),
}));

vi.mock("./agent_info", () => ({
  default: ({ agentId, onClose }: { agentId: string; onClose: () => void }) => (
    <div>
      <p data-testid="agent-info">{agentId}</p>
      <button onClick={onClose}>Close agent</button>
    </div>
  ),
}));

describe("AgentsPanel", () => {
  beforeEach(() => {
    // mockReset (not mockClear) so an unconsumed *Once queue cannot leak into the next test
    vi.mocked(networking.getAgentsList).mockReset().mockResolvedValue({ agents: [] });
    vi.mocked(networking.deleteAgentCall).mockReset().mockResolvedValue({});
  });

  it("should render the Agents panel title", () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(screen.getByText("Agents")).toBeInTheDocument();
  });

  it("should show Add New Agent button for admin users", () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(screen.getByText("Add New Agent")).toBeInTheDocument();
  });

  it("should show Add New Agent button for proxy_admin users", () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="proxy_admin" />);
    expect(screen.getByText("Add New Agent")).toBeInTheDocument();
  });

  it("should not show Add New Agent button for internal_user role", () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Internal User" />);
    expect(screen.queryByText("Add New Agent")).not.toBeInTheDocument();
  });

  it("should not show Add New Agent button for internal_user_viewer role", () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Internal Viewer" />);
    expect(screen.queryByText("Add New Agent")).not.toBeInTheDocument();
  });

  it("should show the Actions column for admin role", async () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(await screen.findByRole("columnheader", { name: /actions/i })).toBeInTheDocument();
  });

  it("should not show the Actions column for internal user role", async () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Internal User" />);
    await waitFor(() => {
      expect(screen.queryByRole("columnheader", { name: /actions/i })).not.toBeInTheDocument();
      expect(screen.getByRole("table")).toBeInTheDocument();
    });
  });

  it("should render the Health Check toggle for admins and non-admins", () => {
    const { unmount } = renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(screen.getByText("Health Check")).toBeInTheDocument();
    unmount();

    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Internal User" />);
    expect(screen.getByText("Health Check")).toBeInTheDocument();
  });

  it("should call getAgentsList with health_check=false on initial load", async () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    await waitFor(() => {
      expect(networking.getAgentsList).toHaveBeenCalledWith("test-token", false);
    });
  });

  it("should show Active when an agent has keys and Needs Setup when it has none", async () => {
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [
        {
          agent_id: "agent-with-key",
          agent_name: "Keyed Agent",
          litellm_params: { model: "gpt-4" },
          spend: 0,
          keys: [{ token: "hash-aaa", key_alias: "primary", key_name: "sk-...aaa" }],
        },
        {
          agent_id: "agent-no-key",
          agent_name: "Keyless Agent",
          litellm_params: { model: "gpt-4" },
          spend: 0,
          keys: [],
        },
      ],
    });

    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);

    const keyedRow = (await screen.findByText("Keyed Agent")).closest("tr")!;
    const keylessRow = screen.getByText("Keyless Agent").closest("tr")!;
    expect(within(keyedRow).getByText("Active")).toBeInTheDocument();
    expect(within(keylessRow).getByText("Needs Setup")).toBeInTheDocument();
  });

  it("should refetch with health_check=true when the toggle is enabled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    await waitFor(() => {
      expect(networking.getAgentsList).toHaveBeenCalledWith("test-token", false);
    });

    await user.click(screen.getByRole("switch"));

    await waitFor(() => {
      expect(networking.getAgentsList).toHaveBeenCalledWith("test-token", true);
    });
  });

  it("should delete an agent through the ⋯ menu and confirm modal, then refetch", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [
        {
          agent_id: "agent-9",
          agent_name: "Doomed Agent",
          litellm_params: { model: "gpt-4" },
          spend: 0,
          keys: [],
        },
      ],
    });

    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);

    await user.click(await screen.findByTestId("agent-actions-agent-9"));
    await user.click(await screen.findByTestId("agent-action-delete"));

    const confirmPrompt = await screen.findByText(/are you sure you want to delete agent: Doomed Agent\?/i);
    const confirmDialog = confirmPrompt.closest('[role="dialog"],[role="alertdialog"]') as HTMLElement;
    await user.click(within(confirmDialog).getByRole("button", { name: /^delete$/i }));

    await waitFor(() => {
      expect(networking.deleteAgentCall).toHaveBeenCalledWith("test-token", "agent-9");
    });
    // one initial load + one post-delete refetch
    await waitFor(() => {
      expect(networking.getAgentsList).toHaveBeenCalledTimes(2);
    });
  });

  it("keeps the delete dialog open until the refreshed list without the deleted agent has loaded", async () => {
    const user = userEvent.setup();
    const doomedAgent = {
      agent_id: "agent-9",
      agent_name: "Doomed Agent",
      litellm_params: { model: "gpt-4" },
      spend: 0,
      keys: [],
    };
    let resolveReload: (value: { agents: never[] }) => void = () => {};
    vi.mocked(networking.getAgentsList)
      .mockResolvedValueOnce({ agents: [doomedAgent] })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveReload = resolve;
          }),
      );
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: "?health_check=true",
    });

    await user.click(await screen.findByTestId("agent-actions-agent-9"));
    await user.click(await screen.findByTestId("agent-action-delete"));
    const confirmDialog = await screen.findByRole("alertdialog");
    await user.click(within(confirmDialog).getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(networking.getAgentsList).toHaveBeenCalledTimes(2));
    expect(networking.getAgentsList).toHaveBeenLastCalledWith("test-token", true);
    expect(screen.getByText(/are you sure you want to delete agent: Doomed Agent\?/i)).toBeInTheDocument();

    await act(async () => {
      resolveReload({ agents: [] });
    });

    await waitFor(() => expect(screen.queryByText(/are you sure you want to delete agent/i)).not.toBeInTheDocument());
    expect(screen.queryByText("Doomed Agent")).not.toBeInTheDocument();
  });

  it("reloads the list with the URL's health_check after an agent is added", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: "?health_check=true",
    });
    await waitFor(() => expect(networking.getAgentsList).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "Finish add" }));

    await waitFor(() => expect(networking.getAgentsList).toHaveBeenCalledTimes(2));
    expect(networking.getAgentsList).toHaveBeenLastCalledWith("test-token", true);
  });

  it("should show a loading skeleton on initial load and clear it once agents arrive", async () => {
    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();
    });
  });

  it("should clear the loading state when there is no access token rather than skeleton forever", async () => {
    renderWithProviders(<AgentsPanel accessToken={null} userRole="Admin" />);
    await waitFor(() => {
      expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();
    });
    expect(screen.getByText("No agents yet")).toBeInTheDocument();
    expect(networking.getAgentsList).not.toHaveBeenCalled();
  });

  it("should not show rows fetched with a previous access token after the token changes", async () => {
    const agentFor = (name: string) => ({
      agent_id: `id-${name}`,
      agent_name: name,
      litellm_params: { model: "gpt-4" },
      spend: 0,
      keys: [],
    });
    let resolveSecond: (value: { agents: ReturnType<typeof agentFor>[] }) => void = () => {};
    vi.mocked(networking.getAgentsList)
      .mockResolvedValueOnce({ agents: [agentFor("first-token-agent")] })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSecond = resolve;
          }),
      );

    const { rerender } = renderWithProviders(<AgentsPanel accessToken="token-a" userRole="Admin" />);
    expect(await screen.findByText("first-token-agent")).toBeInTheDocument();

    rerender(<AgentsPanel accessToken="token-b" userRole="Admin" />);

    // the previous token's rows must not linger while the new token loads
    expect(screen.queryByText("first-token-agent")).not.toBeInTheDocument();
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);

    await act(async () => {
      resolveSecond({ agents: [agentFor("second-token-agent")] });
    });
    expect(await screen.findByText("second-token-agent")).toBeInTheDocument();
  });

  it("should drop previous rows when the fetch for a new token fails", async () => {
    vi.mocked(networking.getAgentsList)
      .mockResolvedValueOnce({
        agents: [
          { agent_id: "stale", agent_name: "Stale Agent", litellm_params: { model: "gpt-4" }, spend: 0, keys: [] },
        ],
      })
      .mockRejectedValueOnce(new Error("unauthorized"));

    const { rerender } = renderWithProviders(<AgentsPanel accessToken="token-a" userRole="Admin" />);
    expect(await screen.findByText("Stale Agent")).toBeInTheDocument();

    rerender(<AgentsPanel accessToken="token-b" userRole="Admin" />);

    await waitFor(() => {
      expect(screen.getByText("No agents yet")).toBeInTheDocument();
    });
    expect(screen.queryByText("Stale Agent")).not.toBeInTheDocument();
  });

  it("should ignore a superseded response so it cannot overwrite the current token's rows", async () => {
    let resolveFirst: (value: {
      agents: { agent_id: string; agent_name: string; litellm_params: { model: string }; spend: number; keys: [] }[];
    }) => void = () => {};
    vi.mocked(networking.getAgentsList)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockResolvedValueOnce({
        agents: [
          { agent_id: "current", agent_name: "Current Agent", litellm_params: { model: "gpt-4" }, spend: 0, keys: [] },
        ],
      });

    const { rerender } = renderWithProviders(<AgentsPanel accessToken="token-a" userRole="Admin" />);
    rerender(<AgentsPanel accessToken="token-b" userRole="Admin" />);

    expect(await screen.findByText("Current Agent")).toBeInTheDocument();

    // the slow token-a response lands last and must be discarded
    await act(async () => {
      resolveFirst({
        agents: [
          { agent_id: "stale", agent_name: "Superseded Agent", litellm_params: { model: "gpt-4" }, spend: 0, keys: [] },
        ],
      });
    });

    expect(screen.queryByText("Superseded Agent")).not.toBeInTheDocument();
    expect(screen.getByText("Current Agent")).toBeInTheDocument();
  });

  it("should keep rows visible during a health-check refetch instead of re-showing the skeleton", async () => {
    const user = userEvent.setup();
    const agents = [
      {
        agent_id: "agent-1",
        agent_name: "Stable Agent",
        litellm_params: { model: "gpt-4" },
        spend: 0,
        keys: [],
      },
    ];
    let resolveRefetch: (value: { agents: typeof agents }) => void = () => {};
    vi.mocked(networking.getAgentsList)
      .mockResolvedValueOnce({ agents })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRefetch = resolve;
          }),
      );

    renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />);
    expect(await screen.findByText("Stable Agent")).toBeInTheDocument();

    await user.click(screen.getByRole("switch"));

    expect(screen.getByText("Stable Agent")).toBeInTheDocument();
    expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();

    await act(async () => {
      resolveRefetch({ agents });
    });
  });

  describe("URL state", () => {
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0];
    const listedAgent = {
      agent_id: "agent-1",
      agent_name: "Listed Agent",
      litellm_params: { model: "gpt-4" },
      spend: 0,
      keys: [],
    };

    it("loads with health_check on when the URL asks for it", async () => {
      renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />, {
        searchParams: "?health_check=true",
      });

      await waitFor(() => expect(networking.getAgentsList).toHaveBeenCalledWith("test-token", true));
      expect(networking.getAgentsList).not.toHaveBeenCalledWith("test-token", false);
      expect(screen.getByRole("switch")).toBeChecked();
    });

    it("writes health_check when the toggle flips and drops it when turned off", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />, { onUrlUpdate });
      await waitFor(() => expect(networking.getAgentsList).toHaveBeenCalledWith("test-token", false));

      await user.click(screen.getByRole("switch"));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("health_check")).toBe("true"));
      await waitFor(() => expect(screen.getByRole("switch")).toBeEnabled());
      expect(screen.getByRole("switch")).toBeChecked();

      await user.click(screen.getByRole("switch"));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("health_check")).toBe(false));
      await waitFor(() => expect(vi.mocked(networking.getAgentsList).mock.calls).toHaveLength(3));
      expect(vi.mocked(networking.getAgentsList).mock.calls.at(-1)).toEqual(["test-token", false]);
    });

    it("renders the detail view for the agent in the URL", () => {
      renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />, {
        searchParams: "?agent=agent-42",
      });

      expect(screen.getByTestId("agent-info")).toHaveTextContent("agent-42");
      expect(screen.queryByRole("table")).not.toBeInTheDocument();
    });

    it("pushes agent when an agent is opened from the table and drops leftover tab and key", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      vi.mocked(networking.getAgentsList).mockResolvedValue({ agents: [listedAgent] });
      renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />, {
        searchParams: "?agent_search=listed&tab=settings&key=stale",
        onUrlUpdate,
      });

      await user.click(await screen.findByText("agent-1"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("agent")).toBe("agent-1"));
      expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("agent_search")).toBe("listed");
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("key")).toBe(false);
      expect(screen.getByTestId("agent-info")).toHaveTextContent("agent-1");
    });

    it("clears agent and its nested detail keys when the detail closes", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />, {
        searchParams: "?agent=agent-1&tab=settings&key=hash-1&health_check=true",
        onUrlUpdate,
      });

      await user.click(screen.getByRole("button", { name: "Close agent" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("agent")).toBe(false));
      const update = lastUrlUpdate(onUrlUpdate);
      expect(update?.searchParams.has("tab")).toBe(false);
      expect(update?.searchParams.has("key")).toBe(false);
      expect(update?.searchParams.get("health_check")).toBe("true");
      expect(update?.options.history).toBe("push");
      expect(await screen.findByRole("table")).toBeInTheDocument();
    });

    it("closes an open detail when Add New Agent is clicked", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<AgentsPanel accessToken="test-token" userRole="Admin" />, {
        searchParams: "?agent=agent-1&tab=settings",
        onUrlUpdate,
      });

      await user.click(screen.getByRole("button", { name: /Add New Agent/ }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("agent")).toBe(false));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
      expect(screen.queryByTestId("agent-info")).not.toBeInTheDocument();
    });
  });
});
