import React from "react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { MCPToolsetsTab } from "./MCPToolsetsTab";
import { fireEvent, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";
import * as networking from "@/components/networking";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { MCPToolset } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({
  createMCPToolset: vi.fn(),
  updateMCPToolset: vi.fn(),
  deleteMCPToolset: vi.fn(),
  listMCPTools: vi.fn(),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({ useMCPToolsets: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({ useMCPServers: vi.fn() }));

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const renderTab = (toolsets: MCPToolset[] = [], searchParams = "", onUrlUpdate?: OnUrlUpdateFunction) => {
  vi.mocked(useMCPToolsets).mockReturnValue({
    data: toolsets,
    isLoading: false,
  } as unknown as ReturnType<typeof useMCPToolsets>);
  vi.mocked(useMCPServers).mockReturnValue({ data: [] } as unknown as ReturnType<typeof useMCPServers>);
  renderWithProviders(<MCPToolsetsTab accessToken="sk-test" userRole="Admin" />, { searchParams, onUrlUpdate });
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
