import React from "react";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { MCPToolsetsTab } from "./MCPToolsetsTab";
import * as networking from "@/components/networking";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { mcpServerToolKeys } from "@/app/(dashboard)/hooks/mcpServers/useMCPServerTools";
import { MCPToolset } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({
  createMCPToolset: vi.fn(),
  updateMCPToolset: vi.fn(),
  deleteMCPToolset: vi.fn(),
  listMCPTools: vi.fn(),
  getProxyBaseUrl: vi.fn().mockReturnValue("https://backend.example.com"),
  getGatewayBaseUrl: vi.fn().mockReturnValue("https://gateway.example.com"),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({ useMCPToolsets: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({ useMCPServers: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "sk-test" }) }));

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const renderInProviders = (queryClient: QueryClient, searchParams = "", onUrlUpdate?: OnUrlUpdateFunction) =>
  render(
    <NuqsTestingAdapter searchParams={searchParams} onUrlUpdate={onUrlUpdate} hasMemory>
      <QueryClientProvider client={queryClient}>
        <MCPToolsetsTab accessToken="sk-test" userRole="Admin" />
      </QueryClientProvider>
    </NuqsTestingAdapter>,
  );

const newQueryClient = () => new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });

const renderTab = (toolsets: MCPToolset[] = [], searchParams = "", onUrlUpdate?: OnUrlUpdateFunction) => {
  vi.mocked(useMCPToolsets).mockReturnValue({
    data: toolsets,
    isLoading: false,
  } as unknown as ReturnType<typeof useMCPToolsets>);
  vi.mocked(useMCPServers).mockReturnValue({ data: [] } as unknown as ReturnType<typeof useMCPServers>);
  renderInProviders(newQueryClient(), searchParams, onUrlUpdate);
};

const dialogWithButton = async (name: string) => {
  const button = await screen.findByRole("button", { name });
  const dialog = button.closest('[role="dialog"]');
  if (dialog === null) {
    throw new Error(`no dialog contains a "${name}" button`);
  }
  return within(dialog as HTMLElement);
};

const openEditFor = async (user: ReturnType<typeof setup>) => {
  await user.click(await screen.findByRole("button", { name: "Open toolset actions" }));
  await user.click(await screen.findByRole("menuitem", { name: "Edit" }));
  return dialogWithButton("Save Changes");
};

const openCreate = async (user: ReturnType<typeof setup>) => {
  await user.click(screen.getByRole("button", { name: /new toolset/i }));
  return dialogWithButton("Create Toolset");
};

describe("MCPToolsetsTab create/edit toolset form", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates a toolset with the typed name, description and no tools", async () => {
    const user = setup();
    vi.mocked(networking.createMCPToolset).mockResolvedValue(
      {} as Awaited<ReturnType<typeof networking.createMCPToolset>>,
    );
    renderTab();

    const dialog = await openCreate(user);
    fireEvent.change(dialog.getByPlaceholderText("e.g. github-linear-tools"), {
      target: { value: "github-linear-tools" },
    });
    fireEvent.change(dialog.getByPlaceholderText("Optional description"), { target: { value: "tools for triage" } });
    await user.click(dialog.getByRole("button", { name: "Create Toolset" }));

    await waitFor(() => {
      expect(networking.createMCPToolset).toHaveBeenCalledWith("sk-test", {
        toolset_name: "github-linear-tools",
        description: "tools for triage",
        tools: [],
      });
    });
    expect(networking.createMCPToolset).toHaveBeenCalledTimes(1);
  });

  it("sends an empty string when the description is left untouched", async () => {
    const user = setup();
    vi.mocked(networking.createMCPToolset).mockResolvedValue(
      {} as Awaited<ReturnType<typeof networking.createMCPToolset>>,
    );
    renderTab();

    const dialog = await openCreate(user);
    fireEvent.change(dialog.getByPlaceholderText("e.g. github-linear-tools"), { target: { value: "solo" } });
    await user.click(dialog.getByRole("button", { name: "Create Toolset" }));

    await waitFor(() => {
      expect(networking.createMCPToolset).toHaveBeenCalledWith("sk-test", {
        toolset_name: "solo",
        description: "",
        tools: [],
      });
    });
  });

  it("blocks the submit and shows the required message when the name is empty", async () => {
    const user = setup();
    renderTab();

    const dialog = await openCreate(user);
    await user.click(dialog.getByRole("button", { name: "Create Toolset" }));

    expect(await dialog.findByText("Please enter a toolset name")).toBeInTheDocument();
    expect(networking.createMCPToolset).not.toHaveBeenCalled();
  });

  it("does not treat a whitespace-only description as absent", async () => {
    const user = setup();
    vi.mocked(networking.createMCPToolset).mockResolvedValue(
      {} as Awaited<ReturnType<typeof networking.createMCPToolset>>,
    );
    renderTab();

    const dialog = await openCreate(user);
    fireEvent.change(dialog.getByPlaceholderText("e.g. github-linear-tools"), { target: { value: "spaced" } });
    fireEvent.change(dialog.getByPlaceholderText("Optional description"), { target: { value: "  " } });
    await user.click(dialog.getByRole("button", { name: "Create Toolset" }));

    await waitFor(() => {
      expect(networking.createMCPToolset).toHaveBeenCalledWith("sk-test", {
        toolset_name: "spaced",
        description: "  ",
        tools: [],
      });
    });
  });

  it("seeds the edit form from the toolset and updates it by id", async () => {
    const user = setup();
    vi.mocked(networking.updateMCPToolset).mockResolvedValue(
      {} as Awaited<ReturnType<typeof networking.updateMCPToolset>>,
    );
    const toolset = {
      toolset_id: "ts-1",
      toolset_name: "existing",
      description: "old description",
      tools: [{ server_id: "srv-1", tool_name: "search" }],
    } as MCPToolset;
    renderTab([toolset]);

    const dialog = await openEditFor(user);
    const name = await dialog.findByDisplayValue("existing");
    expect(name).toBe(dialog.getByPlaceholderText("e.g. github-linear-tools"));
    expect(dialog.getByText("Toolset Name")).toBeInTheDocument();
    expect(dialog.getByPlaceholderText("Optional description")).toHaveValue("old description");

    await user.clear(name);
    fireEvent.change(name, { target: { value: "renamed" } });
    await user.click(dialog.getByRole("button", { name: "Save Changes" }));

    const expectedUpdate = {
      toolset_id: "ts-1",
      toolset_name: "renamed",
      description: "old description",
      tools: [{ server_id: "srv-1", tool_name: "search" }],
    };
    await waitFor(() => {
      expect(networking.updateMCPToolset).toHaveBeenCalledWith("sk-test", expectedUpdate);
    });
  });

  it("seeds an absent description as an empty string rather than failing", async () => {
    const user = setup();
    vi.mocked(networking.updateMCPToolset).mockResolvedValue(
      {} as Awaited<ReturnType<typeof networking.updateMCPToolset>>,
    );
    const toolset = {
      toolset_id: "ts-2",
      toolset_name: "no-desc",
      description: null,
      tools: [],
    } as unknown as MCPToolset;
    renderTab([toolset]);

    const dialog = await openEditFor(user);
    await dialog.findByDisplayValue("no-desc");
    expect(dialog.getByPlaceholderText("Optional description")).toHaveValue("");

    await user.click(dialog.getByRole("button", { name: "Save Changes" }));

    const expectedUpdate = { toolset_id: "ts-2", toolset_name: "no-desc", description: "", tools: [] };
    await waitFor(() => {
      expect(networking.updateMCPToolset).toHaveBeenCalledWith("sk-test", expectedUpdate);
    });
  });
});

describe("MCPToolsetsTab tool search and bulk selection", () => {
  const servers = [
    { server_id: "gh", alias: "github", server_name: "GitHub" },
    { server_id: "lin", alias: "linear", server_name: "Linear" },
    { server_id: "broken", alias: "flaky", server_name: "Flaky" },
  ];
  const catalogs: Record<string, { name: string; description: string }[]> = {
    gh: [
      { name: "create_issue", description: "Open a GitHub issue" },
      { name: "list_pulls", description: "List pull requests" },
    ],
    lin: [
      { name: "create_ticket", description: "Create a Linear issue" },
      { name: "list_projects", description: "List projects" },
    ],
  };

  const renderWithServers = (toolsets: MCPToolset[] = []) => {
    vi.mocked(useMCPToolsets).mockReturnValue({
      data: toolsets,
      isLoading: false,
    } as unknown as ReturnType<typeof useMCPToolsets>);
    vi.mocked(useMCPServers).mockReturnValue({ data: servers, isSuccess: true } as unknown as ReturnType<
      typeof useMCPServers
    >);
    const queryClient = newQueryClient();
    renderInProviders(queryClient);
    return queryClient;
  };

  const editToolset = (tools: MCPToolset["tools"]) =>
    ({ toolset_id: "ts-1", toolset_name: "triage", description: "", tools }) as MCPToolset;

  const expectedUpdate = (tools: MCPToolset["tools"]) => ({
    toolset_id: "ts-1",
    toolset_name: "triage",
    description: "",
    tools,
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.listMCPTools).mockImplementation(async (_token, serverId) =>
      serverId in catalogs
        ? { tools: catalogs[serverId], error: null }
        : { tools: [], error: "http_502", message: "upstream down" },
    );
    vi.mocked(networking.createMCPToolset).mockResolvedValue(
      {} as Awaited<ReturnType<typeof networking.createMCPToolset>>,
    );
    vi.mocked(networking.updateMCPToolset).mockResolvedValue(
      {} as Awaited<ReturnType<typeof networking.updateMCPToolset>>,
    );
  });

  it("searches tool names and descriptions across collapsed servers and selects every match", async () => {
    const user = setup();
    renderWithServers();

    await openCreate(user);
    fireEvent.change(screen.getByPlaceholderText("e.g. github-linear-tools"), { target: { value: "issues" } });
    await user.type(screen.getByRole("textbox", { name: "Search tools or servers" }), "issue");

    expect(await screen.findByRole("checkbox", { name: /^create_ticket/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /^create_issue/ })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /^list_pulls/ })).not.toBeInTheDocument();
    expect(await screen.findByText("2 matching tools")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Select all matching" }));
    await user.click(screen.getByRole("button", { name: "Create Toolset" }));

    await waitFor(() => {
      expect(networking.createMCPToolset).toHaveBeenCalledWith("sk-test", {
        toolset_name: "issues",
        description: "",
        tools: [
          { server_id: "gh", tool_name: "create_issue" },
          { server_id: "lin", tool_name: "create_ticket" },
        ],
      });
    });
  });

  it("deselecting all matches keeps selected tools the search hides", async () => {
    const user = setup();
    renderWithServers([
      editToolset([
        { server_id: "gh", tool_name: "list_pulls" },
        { server_id: "gh", tool_name: "create_issue" },
        { server_id: "lin", tool_name: "create_ticket" },
      ]),
    ]);

    await openEditFor(user);
    await user.type(screen.getByRole("textbox", { name: "Search tools or servers" }), "create");
    expect(await screen.findByRole("checkbox", { name: /^create_ticket/ })).toBeChecked();

    await user.click(screen.getByRole("button", { name: "Deselect all matching" }));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(networking.updateMCPToolset).toHaveBeenCalledWith(
        "sk-test",
        expectedUpdate([{ server_id: "gh", tool_name: "list_pulls" }]),
      );
    });
  });

  it("selects all of one server's tools from its header and toggles single rows, fetching only that server", async () => {
    const user = setup();
    renderWithServers();

    await openCreate(user);
    fireEvent.change(screen.getByPlaceholderText("e.g. github-linear-tools"), { target: { value: "gh-only" } });
    await user.click(screen.getByRole("button", { name: /^github/ }));
    expect(await screen.findByRole("checkbox", { name: /^list_pulls/ })).toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: "Select all tools on github" }));
    expect(screen.getByRole("checkbox", { name: /^list_pulls/ })).toBeChecked();
    // Clicking anywhere on a tool row toggles it exactly once.
    await user.click(screen.getByText("List pull requests"));
    expect(screen.getByRole("checkbox", { name: /^list_pulls/ })).not.toBeChecked();
    await user.click(screen.getByRole("button", { name: "Create Toolset" }));

    await waitFor(() => {
      expect(networking.createMCPToolset).toHaveBeenCalledWith("sk-test", {
        toolset_name: "gh-only",
        description: "",
        tools: [{ server_id: "gh", tool_name: "create_issue" }],
      });
    });
    expect(networking.listMCPTools).toHaveBeenCalledTimes(1);
  });

  it("keeps and flags tools whose server is unreachable or gone when saving", async () => {
    const user = setup();
    const tools = [
      { server_id: "broken", tool_name: "flaky_tool" },
      { server_id: "deleted", tool_name: "old_tool" },
      { server_id: "gh", tool_name: "create_issue" },
    ];
    renderWithServers([editToolset(tools)]);

    await openEditFor(user);
    await user.type(screen.getByRole("textbox", { name: "Search tools or servers" }), "create");

    expect(await screen.findByText("Couldn't load tools for this server.")).toBeInTheDocument();
    expect(await screen.findByText("server unreachable")).toBeInTheDocument();
    expect(screen.getByText("server not found")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(networking.updateMCPToolset).toHaveBeenCalledWith("sk-test", expectedUpdate(tools));
    });
  });

  it("stops fetching server tools once the editor is closed after a search", async () => {
    const user = setup();
    const queryClient = renderWithServers();

    await openCreate(user);
    await user.type(screen.getByRole("textbox", { name: "Search tools or servers" }), "issue");
    expect(await screen.findByRole("checkbox", { name: /^create_ticket/ })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    const callsWhileOpen = vi.mocked(networking.listMCPTools).mock.calls.length;

    await act(() => queryClient.invalidateQueries({ queryKey: mcpServerToolKeys.all }));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(networking.listMCPTools).toHaveBeenCalledTimes(callsWhileOpen);

    await openCreate(user);
    expect(screen.getByRole("textbox", { name: "Search tools or servers" })).toHaveValue("");
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(networking.listMCPTools).toHaveBeenCalledTimes(callsWhileOpen);
  });

  it("does not refetch a previously expanded server when the editor is reopened", async () => {
    const user = setup();
    const queryClient = renderWithServers();

    await openCreate(user);
    await user.click(screen.getByRole("button", { name: /^github/ }));
    expect(await screen.findByRole("checkbox", { name: /^list_pulls/ })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await act(() => queryClient.invalidateQueries({ queryKey: mcpServerToolKeys.all }));

    await openCreate(user);
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(networking.listMCPTools).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("checkbox", { name: /^list_pulls/ })).not.toBeInTheDocument();
  });

  it("clears the whole toolset with Clear all", async () => {
    const user = setup();
    renderWithServers([editToolset([{ server_id: "gh", tool_name: "create_issue" }])]);

    await openEditFor(user);
    await user.click(screen.getByRole("button", { name: "Clear all" }));
    expect(screen.getByText("No tools added yet")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(networking.updateMCPToolset).toHaveBeenCalledWith("sk-test", expectedUpdate([]));
    });
  });
});

describe("MCPToolsetsTab usage guide", () => {
  it("shows the toolset URL on the gateway host", () => {
    renderTab();

    expect(screen.getByText(/https:\/\/gateway\.example\.com\/toolset\/<toolset-name>\/mcp/)).toBeInTheDocument();
  });
});

const makeToolset = (name: string, day: number): MCPToolset =>
  ({
    toolset_id: `id-${name}`,
    toolset_name: name,
    description: "",
    tools: [],
    created_at: `2024-01-${String(day).padStart(2, "0")}T00:00:00Z`,
  }) as MCPToolset;

const shownToolsetNames = () => screen.getAllByText(/^ts-\d{2}$/).map((node) => node.textContent);

describe("MCPToolsetsTab table URL state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const namedToolsets = [makeToolset("ts-02", 1), makeToolset("ts-03", 2), makeToolset("ts-01", 3)];
  const manyToolsets = Array.from({ length: 30 }, (_, index) =>
    makeToolset(`ts-${String(index + 1).padStart(2, "0")}`, index + 1),
  );

  it("lists the newest toolsets first by default", () => {
    renderTab(namedToolsets);

    expect(shownToolsetNames()).toEqual(["ts-01", "ts-03", "ts-02"]);
  });

  it("orders the rows by the sort in the URL", () => {
    renderTab(namedToolsets, "?toolset_sort_by=toolset_name&toolset_sort_order=asc");

    expect(shownToolsetNames()).toEqual(["ts-01", "ts-02", "ts-03"]);
  });

  it("writes the sort the user picks under the toolset prefix", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderTab(namedToolsets, "?toolset_page=1", onUrlUpdate);

    await userEvent.click(screen.getByTestId("sort-header-toolset_name"));

    const update = onUrlUpdate.mock.calls.at(-1)?.[0];
    expect(update?.searchParams.get("toolset_sort_by")).toBe("toolset_name");
    expect(update?.searchParams.get("toolset_sort_order")).toBe("asc");
    expect(update?.searchParams.has("sort_by")).toBe(false);
    expect(shownToolsetNames()).toEqual(["ts-01", "ts-02", "ts-03"]);
  });

  it("opens the page in the URL", () => {
    renderTab(manyToolsets, "?toolset_page=2");

    expect(screen.getByText("Page 2 of 2")).toBeInTheDocument();
    expect(shownToolsetNames()).toEqual(["ts-05", "ts-04", "ts-03", "ts-02", "ts-01"]);
  });

  it("writes the page the user moves to under the toolset prefix", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderTab(manyToolsets, "", onUrlUpdate);

    expect(shownToolsetNames()).toHaveLength(25);
    await userEvent.click(screen.getByRole("button", { name: "Go to next page" }));

    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("toolset_page")).toBe("2");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("page")).toBe(false);
    expect(shownToolsetNames()).toEqual(["ts-05", "ts-04", "ts-03", "ts-02", "ts-01"]);
  });
});
