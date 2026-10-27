import React from "react";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
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

const renderTab = (toolsets: MCPToolset[] = []) => {
  vi.mocked(useMCPToolsets).mockReturnValue({
    data: toolsets,
    isLoading: false,
  } as unknown as ReturnType<typeof useMCPToolsets>);
  vi.mocked(useMCPServers).mockReturnValue({ data: [] } as unknown as ReturnType<typeof useMCPServers>);
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>
      <MCPToolsetsTab accessToken="sk-test" userRole="Admin" />
    </QueryClientProvider>,
  );
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
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    render(
      <QueryClientProvider client={queryClient}>
        <MCPToolsetsTab accessToken="sk-test" userRole="Admin" />
      </QueryClientProvider>,
    );
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

    const dialog = await openCreate(user);
    fireEvent.change(dialog.getByPlaceholderText("e.g. github-linear-tools"), { target: { value: "issues" } });
    await user.type(dialog.getByRole("textbox", { name: "Search tools or servers" }), "issue");

    expect(await dialog.findByRole("checkbox", { name: /^create_ticket/ })).toBeInTheDocument();
    expect(dialog.getByRole("checkbox", { name: /^create_issue/ })).toBeInTheDocument();
    expect(dialog.queryByRole("checkbox", { name: /^list_pulls/ })).not.toBeInTheDocument();
    expect(await dialog.findByText("2 matching tools")).toBeInTheDocument();

    await user.click(dialog.getByRole("button", { name: "Select all matching" }));
    await user.click(dialog.getByRole("button", { name: "Create Toolset" }));

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

    const dialog = await openEditFor(user);
    await user.type(dialog.getByRole("textbox", { name: "Search tools or servers" }), "create");
    expect(await dialog.findByRole("checkbox", { name: /^create_ticket/ })).toBeChecked();

    await user.click(dialog.getByRole("button", { name: "Deselect all matching" }));
    await user.click(dialog.getByRole("button", { name: "Save Changes" }));

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

    const dialog = await openCreate(user);
    fireEvent.change(dialog.getByPlaceholderText("e.g. github-linear-tools"), { target: { value: "gh-only" } });
    await user.click(dialog.getByRole("button", { name: /^github/ }));
    expect(await dialog.findByRole("checkbox", { name: /^list_pulls/ })).toBeInTheDocument();

    await user.click(dialog.getByRole("checkbox", { name: "Select all tools on github" }));
    expect(dialog.getByRole("checkbox", { name: /^list_pulls/ })).toBeChecked();
    // Clicking anywhere on a tool row toggles it exactly once.
    await user.click(dialog.getByText("List pull requests"));
    expect(dialog.getByRole("checkbox", { name: /^list_pulls/ })).not.toBeChecked();
    await user.click(dialog.getByRole("button", { name: "Create Toolset" }));

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

    const dialog = await openEditFor(user);
    await user.type(dialog.getByRole("textbox", { name: "Search tools or servers" }), "create");

    expect(await dialog.findByText("Couldn't load tools for this server.")).toBeInTheDocument();
    expect(await dialog.findByText("server unreachable")).toBeInTheDocument();
    expect(dialog.getByText("server not found")).toBeInTheDocument();

    await user.click(dialog.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(networking.updateMCPToolset).toHaveBeenCalledWith("sk-test", expectedUpdate(tools));
    });
  });

  it("stops fetching server tools once the editor is closed after a search", async () => {
    const user = setup();
    const queryClient = renderWithServers();

    const dialog = await openCreate(user);
    await user.type(dialog.getByRole("textbox", { name: "Search tools or servers" }), "issue");
    expect(await dialog.findByRole("checkbox", { name: /^create_ticket/ })).toBeInTheDocument();
    await user.click(dialog.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    const callsWhileOpen = vi.mocked(networking.listMCPTools).mock.calls.length;

    await act(() => queryClient.invalidateQueries({ queryKey: mcpServerToolKeys.all }));
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(networking.listMCPTools).toHaveBeenCalledTimes(callsWhileOpen);
  });

  it("clears the whole toolset with Clear all", async () => {
    const user = setup();
    renderWithServers([editToolset([{ server_id: "gh", tool_name: "create_issue" }])]);

    const dialog = await openEditFor(user);
    await user.click(dialog.getByRole("button", { name: "Clear all" }));
    expect(dialog.getByText("No tools added yet")).toBeInTheDocument();
    await user.click(dialog.getByRole("button", { name: "Save Changes" }));

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
