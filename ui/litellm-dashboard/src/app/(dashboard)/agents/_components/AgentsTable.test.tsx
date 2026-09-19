import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, it, expect, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";

import AgentsTable from "./AgentsTable";
import { Agent } from "@/components/agents/types";

const baseProps = {
  isLoading: false,
  isAdmin: true,
  healthCheckEnabled: false,
  isHealthCheckLoading: false,
  onHealthCheckToggle: vi.fn(),
  onAgentClick: vi.fn(),
  onDeleteClick: vi.fn(),
};

const makeAgent = (overrides: Partial<Agent> = {}): Agent => ({
  agent_id: "agent-1",
  agent_name: "Test Agent",
  litellm_params: { model: "gpt-4" },
  spend: 0,
  keys: [{ token: "hash-1", key_alias: "primary", key_name: "sk-...1" }],
  created_at: "2023-01-01T00:00:00Z",
  ...overrides,
});

describe("AgentsTable", () => {
  it("renders every column header", () => {
    renderWithProviders(<AgentsTable agents={[]} {...baseProps} />);
    for (const header of ["Agent Name", "Agent ID", "Spend (USD)", "Model", "Created", "Status"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("renders the agent's model and opens the detail view when the ID cell is clicked", async () => {
    const user = userEvent.setup();
    const onAgentClick = vi.fn();
    const agent = makeAgent({ agent_id: "agent-xyz", agent_name: "Router", litellm_params: { model: "claude-3-5" } });
    renderWithProviders(<AgentsTable agents={[agent]} {...baseProps} onAgentClick={onAgentClick} />);

    expect(screen.getByText("claude-3-5")).toBeInTheDocument();

    await user.click(screen.getByText("agent-xyz"));
    expect(onAgentClick).toHaveBeenCalledWith("agent-xyz");
  });

  it("marks agents Active when they have keys and Needs Setup when they have none", () => {
    renderWithProviders(
      <AgentsTable
        agents={[
          makeAgent({ agent_id: "keyed", agent_name: "Keyed Agent", keys: [{ token: "k" }] }),
          makeAgent({ agent_id: "keyless", agent_name: "Keyless Agent", keys: [] }),
        ]}
        {...baseProps}
      />,
    );

    const keyedRow = screen.getByText("Keyed Agent").closest("tr")!;
    const keylessRow = screen.getByText("Keyless Agent").closest("tr")!;
    expect(within(keyedRow).getByText("Active")).toBeInTheDocument();
    expect(within(keylessRow).getByText("Needs Setup")).toBeInTheDocument();
  });

  it("deletes an agent through the ⋯ actions menu", async () => {
    const user = userEvent.setup();
    const onDeleteClick = vi.fn();
    const agent = makeAgent({ agent_id: "agent-9", agent_name: "Doomed Agent" });
    renderWithProviders(<AgentsTable agents={[agent]} {...baseProps} onDeleteClick={onDeleteClick} />);

    await user.click(screen.getByTestId("agent-actions-agent-9"));
    await user.click(await screen.findByTestId("agent-action-delete"));

    expect(onDeleteClick).toHaveBeenCalledWith("agent-9", "Doomed Agent");
  });

  it("filters agents by name or by agent card description", () => {
    renderWithProviders(
      <AgentsTable
        agents={[
          makeAgent({ agent_id: "a1", agent_name: "Billing Router" }),
          makeAgent({
            agent_id: "a2",
            agent_name: "Second Agent",
            agent_card_params: { description: "handles support tickets" },
          }),
        ]}
        {...baseProps}
      />,
    );

    const search = screen.getByPlaceholderText("Search agents by name, ID, or description...");
    fireEvent.change(search, { target: { value: "billing" } });
    expect(screen.getByText("Billing Router")).toBeInTheDocument();
    expect(screen.queryByText("Second Agent")).not.toBeInTheDocument();

    fireEvent.change(search, { target: { value: "support tickets" } });
    expect(screen.getByText("Second Agent")).toBeInTheDocument();
    expect(screen.queryByText("Billing Router")).not.toBeInTheDocument();
  });

  it("filters agents by a pasted agent_id so only that agent's row survives", () => {
    renderWithProviders(
      <AgentsTable
        agents={[
          makeAgent({ agent_id: "5f3c2a1b-9d8e-4f7a-b6c5-d4e3f2a1b0c9", agent_name: "Billing Router" }),
          makeAgent({ agent_id: "0a9b8c7d-6e5f-4a3b-8c2d-1e0f9a8b7c6d", agent_name: "Second Agent" }),
        ]}
        {...baseProps}
      />,
    );

    const search = screen.getByPlaceholderText("Search agents by name, ID, or description...");
    fireEvent.change(search, { target: { value: "5f3c2a1b-9d8e-4f7a-b6c5-d4e3f2a1b0c9" } });
    expect(screen.getByText("Billing Router")).toBeInTheDocument();
    expect(screen.queryByText("Second Agent")).not.toBeInTheDocument();

    fireEvent.change(search, { target: { value: "ffffffff-0000-4000-8000-000000000000" } });
    expect(screen.queryByText("Billing Router")).not.toBeInTheDocument();
    expect(screen.queryByText("Second Agent")).not.toBeInTheDocument();
    expect(screen.getByText("No matching agents")).toBeInTheDocument();
  });

  it("shows the no-match empty state when the search matches nothing", () => {
    renderWithProviders(<AgentsTable agents={[makeAgent()]} {...baseProps} />);

    fireEvent.change(screen.getByPlaceholderText("Search agents by name, ID, or description..."), {
      target: { value: "zzzz" },
    });
    expect(screen.queryByText("Test Agent")).not.toBeInTheDocument();
    expect(screen.getByText("No matching agents")).toBeInTheDocument();
  });

  it("hides the actions column entirely for non-admins", () => {
    const agent = makeAgent({ agent_id: "agent-2" });
    renderWithProviders(<AgentsTable agents={[agent]} {...baseProps} isAdmin={false} />);

    expect(screen.queryByTestId("agent-actions-agent-2")).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: /actions/i })).not.toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("shows the actions column for admins", () => {
    renderWithProviders(<AgentsTable agents={[makeAgent({ agent_id: "agent-3" })]} {...baseProps} isAdmin />);
    expect(screen.getByRole("columnheader", { name: /actions/i })).toBeInTheDocument();
    expect(screen.getByTestId("agent-actions-agent-3")).toBeInTheDocument();
  });

  it("defaults to sorting by created_at descending (newest first)", () => {
    renderWithProviders(
      <AgentsTable
        agents={[
          makeAgent({ agent_id: "old", agent_name: "Alpha Agent", created_at: "2021-06-01T00:00:00Z" }),
          makeAgent({ agent_id: "new", agent_name: "Beta Agent", created_at: "2023-06-01T00:00:00Z" }),
        ]}
        {...baseProps}
      />,
    );

    const bodyRows = screen.getAllByRole("row").slice(1);
    expect(bodyRows[0]).toHaveTextContent(/Beta Agent/);
    expect(bodyRows[1]).toHaveTextContent(/Alpha Agent/);
  });

  it("sorts agents with no created_at last, never ahead of dated ones", () => {
    renderWithProviders(
      <AgentsTable
        agents={[
          makeAgent({ agent_id: "old", agent_name: "Alpha Agent", created_at: "2021-06-01T00:00:00Z" }),
          makeAgent({ agent_id: "undated", agent_name: "Undated Agent", created_at: undefined }),
          makeAgent({ agent_id: "new", agent_name: "Beta Agent", created_at: "2023-06-01T00:00:00Z" }),
        ]}
        {...baseProps}
      />,
    );

    const bodyRows = screen.getAllByRole("row").slice(1);
    expect(bodyRows[0]).toHaveTextContent(/Beta Agent/);
    expect(bodyRows[1]).toHaveTextContent(/Alpha Agent/);
    expect(bodyRows[2]).toHaveTextContent(/Undated Agent/);
  });

  it("shows a rich empty state when there are no agents", () => {
    renderWithProviders(<AgentsTable agents={[]} {...baseProps} />);
    expect(screen.getByText("No agents yet")).toBeInTheDocument();
    expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();
  });

  it("renders loading skeleton rows on initial load instead of the empty state", () => {
    renderWithProviders(<AgentsTable agents={[]} {...baseProps} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No agents yet")).not.toBeInTheDocument();
  });

  it("invokes the health-check toggle from the toolbar", async () => {
    const user = userEvent.setup();
    const onHealthCheckToggle = vi.fn();
    renderWithProviders(<AgentsTable agents={[]} {...baseProps} onHealthCheckToggle={onHealthCheckToggle} />);

    expect(screen.getByText("Health Check")).toBeInTheDocument();
    await user.click(screen.getByRole("switch"));
    expect(onHealthCheckToggle).toHaveBeenCalledWith(true);
  });

  describe("URL table state", () => {
    const searchInput = () => screen.getByPlaceholderText("Search agents by name, ID, or description...");
    const rowNames = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((row) => within(row).getByTitle(/Agent/).textContent);
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0];
    const twoAgents = [
      makeAgent({ agent_id: "b", agent_name: "Beta Agent", created_at: "2021-01-01T00:00:00Z" }),
      makeAgent({ agent_id: "a", agent_name: "Alpha Agent", created_at: "2023-01-01T00:00:00Z" }),
    ];
    const manyAgents = Array.from({ length: 30 }, (_, index) =>
      makeAgent({
        agent_id: `id-${index}`,
        agent_name: `Agent ${String(index).padStart(2, "0")}`,
        created_at: `2024-01-01T00:00:${String(59 - index).padStart(2, "0")}Z`,
      }),
    );

    it("filters rows by the agent_search in the URL", () => {
      renderWithProviders(<AgentsTable agents={twoAgents} {...baseProps} />, { searchParams: "?agent_search=beta" });
      expect(searchInput()).toHaveValue("beta");
      expect(rowNames()).toEqual(["Beta Agent"]);
    });

    it("writes the search to agent_search, resets the page, and drops it when cleared", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<AgentsTable agents={twoAgents} {...baseProps} />, { searchParams: "?page=2", onUrlUpdate });

      fireEvent.change(searchInput(), { target: { value: "alpha" } });

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("agent_search")).toBe("alpha"));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false);
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("search")).toBe(false);
      expect(rowNames()).toEqual(["Alpha Agent"]);

      await user.click(screen.getByRole("button", { name: "Clear search" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("agent_search")).toBe(false));
      expect(rowNames()).toEqual(["Alpha Agent", "Beta Agent"]);
    });

    it("orders rows by the sort in the URL", () => {
      renderWithProviders(<AgentsTable agents={twoAgents} {...baseProps} />, {
        searchParams: "?sort_by=agent_name&sort_order=desc",
      });
      expect(rowNames()).toEqual(["Beta Agent", "Alpha Agent"]);
    });

    it.each(["agent_name", "agent_id", "spend"])(
      "honors sort_by=%s from the URL instead of falling back to created_at",
      (sortBy) => {
        const olderZulu: Agent = {
          ...makeAgent(),
          agent_id: "z-id",
          agent_name: "Zulu Agent",
          spend: 5,
          created_at: "2021-01-01T00:00:00Z",
        };
        const newerAlpha: Agent = {
          ...makeAgent(),
          agent_id: "a-id",
          agent_name: "Alpha Agent",
          spend: 1,
          created_at: "2023-01-01T00:00:00Z",
        };
        renderWithProviders(<AgentsTable agents={[olderZulu, newerAlpha]} {...baseProps} />, {
          searchParams: `?sort_by=${sortBy}&sort_order=asc`,
        });
        expect(rowNames()).toEqual(["Alpha Agent", "Zulu Agent"]);
      },
    );

    it("goes back to the first page when the health check is toggled", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const onHealthCheckToggle = vi.fn();
      renderWithProviders(
        <AgentsTable agents={manyAgents} {...baseProps} onHealthCheckToggle={onHealthCheckToggle} />,
        { searchParams: "?page=2&page_size=10", onUrlUpdate },
      );
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 3");

      await user.click(screen.getByRole("switch"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page_size")).toBe("10");
      expect(onHealthCheckToggle).toHaveBeenCalledWith(true);
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 3");
    });

    it("writes the sort to the URL when a header is clicked", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<AgentsTable agents={twoAgents} {...baseProps} />, { onUrlUpdate });
      expect(rowNames()).toEqual(["Alpha Agent", "Beta Agent"]);

      await user.click(screen.getByTestId("sort-header-agent_id"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_by")).toBe("agent_id"));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_order")).toBe("asc");
      expect(rowNames()).toEqual(["Alpha Agent", "Beta Agent"]);

      await user.click(screen.getByTestId("sort-header-agent_id"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("sort_order")).toBe(false));
      expect(rowNames()).toEqual(["Beta Agent", "Alpha Agent"]);
    });

    it("opens the page and page size named in the URL", () => {
      renderWithProviders(<AgentsTable agents={manyAgents} {...baseProps} />, { searchParams: "?page=3&page_size=10" });
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 3 of 3");
      expect(rowNames()[0]).toBe("Agent 20");
      expect(rowNames()).toHaveLength(10);
    });

    it("writes the page to the URL when paging forward", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<AgentsTable agents={manyAgents} {...baseProps} />, { onUrlUpdate });

      await user.click(screen.getByRole("button", { name: "Go to next page" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page")).toBe("2"));
      expect(rowNames()[0]).toBe("Agent 25");
    });
  });
});
