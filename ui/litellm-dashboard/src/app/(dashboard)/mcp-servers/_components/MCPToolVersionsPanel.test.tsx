import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import type { MCPToolVersion } from "@/components/networking";
import { mcpServersKeys } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { MCPToolVersionsPanel } from "./MCPToolVersionsPanel";

vi.mock("@/components/networking", () => ({
  getMCPToolVersions: vi.fn(),
  pinMCPServerTools: vi.fn(),
  setMCPToolVersionDeprecation: vi.fn(),
  clearMCPToolVersionDeprecation: vi.fn(),
}));

const toolVersions: MCPToolVersion[] = [
  {
    server_id: "srv-1",
    tool_name: "list_notes",
    version: 2,
    description: "List notes",
    input_schema: {},
    change_kind: "breaking",
    changes: [{ breaking: true, summary: 'Parameter "limit" type changed from "string" to "integer"' }],
    changelog: "Updated the limit parameter",
    deprecated_at: "2026-10-02T00:00:00Z",
    sunset_date: "2026-12-31T00:00:00Z",
    deprecation_note: "Move to v3",
    created_at: "2026-10-02T00:00:00Z",
    created_by: "admin",
  },
  {
    server_id: "srv-1",
    tool_name: "list_notes",
    version: 1,
    description: "List notes",
    input_schema: {},
    change_kind: "initial",
    changes: [],
    changelog: null,
    deprecated_at: null,
    sunset_date: null,
    deprecation_note: null,
    created_at: "2026-10-01T00:00:00Z",
    created_by: "admin",
  },
  {
    server_id: "srv-1",
    tool_name: "read_note",
    version: 1,
    description: "Read a note",
    input_schema: {},
    change_kind: "non_breaking",
    changes: [{ breaking: false, summary: "Description changed" }],
    changelog: null,
    deprecated_at: null,
    sunset_date: null,
    deprecation_note: null,
    created_at: "2026-10-01T00:00:00Z",
    created_by: "admin",
  },
];

const renderPanel = (isProxyAdmin = true, customHeaders?: Record<string, string>) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>
      <MCPToolVersionsPanel
        serverId="srv-1"
        accessToken="token"
        isProxyAdmin={isProxyAdmin}
        customHeaders={customHeaders}
      />
    </QueryClientProvider>,
  );

describe("MCPToolVersionsPanel", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.getMCPToolVersions).mockResolvedValue(toolVersions);
    vi.mocked(networking.pinMCPServerTools).mockResolvedValue({});
    vi.mocked(networking.setMCPToolVersionDeprecation).mockResolvedValue(toolVersions[0]);
    vi.mocked(networking.clearMCPToolVersionDeprecation).mockResolvedValue(toolVersions[0]);
  });

  it("shows breaking, non-breaking and deprecated sunset badges", async () => {
    renderPanel();

    expect(await screen.findByText("Breaking")).toBeInTheDocument();
    expect(screen.getAllByText("Breaking").length).toBeGreaterThan(0);
    expect(screen.getByText("Non-breaking")).toBeInTheDocument();
    expect(screen.getByText("Deprecated, sunset 2026-12-31")).toBeInTheDocument();
  });

  it("expands version history newest first with changes and changelog", async () => {
    renderPanel(false);

    const toggle = await screen.findByRole("button", { name: "Show history for list_notes" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);

    expect(screen.getByRole("button", { name: "Hide history for list_notes" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(screen.getByText('Parameter "limit" type changed from "string" to "integer"')).toBeInTheDocument();
    expect(screen.getByText("Updated the limit parameter")).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 4 }).map((heading) => heading.textContent)).toEqual(["v2", "v1"]);
    expect(screen.getByText("Created 2026-10-02")).toBeInTheDocument();
  });

  it("hides pin and deprecation controls from non-admins", async () => {
    renderPanel(false);

    expect(await screen.findByRole("button", { name: "Show history for list_notes" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Pin and record versions" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Show history for list_notes" }));
    expect(screen.queryByRole("button", { name: "Deprecate list_notes v1" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit deprecation for list_notes v2" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Remove deprecation for list_notes v2" })).not.toBeInTheDocument();
  });

  it("sends the changelog when the admin pins tools", async () => {
    vi.mocked(networking.getMCPToolVersions).mockResolvedValue([]);
    renderPanel();

    fireEvent.change(await screen.findByLabelText("Changelog"), {
      target: { value: "  Published first release  " },
    });
    await userEvent.click(screen.getByRole("button", { name: "Pin and record versions" }));

    expect(networking.pinMCPServerTools).toHaveBeenCalledWith(
      "token",
      "srv-1",
      "  Published first release  ",
      undefined,
    );
  });

  it("keeps pinning available when the version history fails to load", async () => {
    vi.mocked(networking.getMCPToolVersions).mockRejectedValue(new Error("history unavailable"));
    renderPanel();

    expect(await screen.findByText("history unavailable")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Pin and record versions" }));

    expect(networking.pinMCPServerTools).toHaveBeenCalledWith("token", "srv-1", "", undefined);
  });

  it("forwards upstream headers when the admin pins tools", async () => {
    vi.mocked(networking.getMCPToolVersions).mockResolvedValue([]);
    const customHeaders: Record<string, string> = {
      "x-mcp-weather-authorization": "Bearer t",
    };
    renderPanel(true, customHeaders);

    await screen.findByLabelText("Changelog");
    await userEvent.click(screen.getByRole("button", { name: "Pin and record versions" }));

    expect(networking.pinMCPServerTools).toHaveBeenCalledWith("token", "srv-1", "", customHeaders);
  });

  it("invalidates all pinned catalog queries after pinning succeeds", async () => {
    vi.mocked(networking.getMCPToolVersions).mockResolvedValue([]);
    const invalidateQueries = vi.spyOn(QueryClient.prototype, "invalidateQueries");
    renderPanel();

    fireEvent.change(await screen.findByLabelText("Changelog"), {
      target: { value: "Initial catalog" },
    });
    await userEvent.click(screen.getByRole("button", { name: "Pin and record versions" }));

    await waitFor(() => expect(invalidateQueries).toHaveBeenCalledTimes(3));
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcpToolVersions", "srv-1"] });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ["mcpTools", "srv-1"] });
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: mcpServersKeys.all });
  });

  it("sends sunset dates at UTC midnight", async () => {
    renderPanel();

    await userEvent.click(await screen.findByRole("button", { name: "Show history for read_note" }));
    await userEvent.click(screen.getByRole("button", { name: "Deprecate read_note v1" }));
    fireEvent.change(screen.getByLabelText("Sunset date"), { target: { value: "2026-12-31" } });
    fireEvent.change(screen.getByLabelText("Note"), { target: { value: "Move to v2" } });
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(networking.setMCPToolVersionDeprecation).toHaveBeenCalledWith(
      "token",
      { serverId: "srv-1", toolName: "read_note", version: 1 },
      {
        sunset_date: "2026-12-31T00:00:00Z",
        deprecation_note: "Move to v2",
      },
    );
  });

  it("prefills and edits an existing deprecation", async () => {
    renderPanel();

    await userEvent.click(await screen.findByRole("button", { name: "Show history for list_notes" }));
    await userEvent.click(screen.getByRole("button", { name: "Edit deprecation for list_notes v2" }));

    expect(screen.getByLabelText("Sunset date")).toHaveValue("2026-12-31");
    expect(screen.getByLabelText("Note")).toHaveValue("Move to v3");

    fireEvent.change(screen.getByLabelText("Sunset date"), { target: { value: "2027-01-15" } });
    fireEvent.change(screen.getByLabelText("Note"), { target: { value: "Use v4" } });
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(networking.setMCPToolVersionDeprecation).toHaveBeenCalledWith(
      "token",
      { serverId: "srv-1", toolName: "list_notes", version: 2 },
      {
        sunset_date: "2027-01-15T00:00:00Z",
        deprecation_note: "Use v4",
      },
    );
  });

  it("reads timestamps without an offset as UTC dates in timezones east of UTC", async () => {
    const originalTimezone = process.env.TZ;
    process.env.TZ = "Asia/Tokyo";
    try {
      vi.mocked(networking.getMCPToolVersions).mockResolvedValue([
        { ...toolVersions[0], sunset_date: "2026-12-31T00:00:00" },
        ...toolVersions.slice(1),
      ]);
      renderPanel();

      expect(await screen.findByText("Deprecated, sunset 2026-12-31")).toBeInTheDocument();
      await userEvent.click(screen.getByRole("button", { name: "Show history for list_notes" }));
      await userEvent.click(screen.getByRole("button", { name: "Edit deprecation for list_notes v2" }));
      expect(screen.getByLabelText("Sunset date")).toHaveValue("2026-12-31");
    } finally {
      process.env.TZ = originalTimezone;
    }
  });

  it("shows loading separately from the empty state", () => {
    vi.mocked(networking.getMCPToolVersions).mockReturnValue(new Promise(() => {}));
    renderPanel();

    expect(screen.getByRole("status", { name: "Loading tool versions" })).toBeInTheDocument();
    expect(
      screen.queryByText("No versions recorded yet. Pin the tool list to start tracking versions."),
    ).not.toBeInTheDocument();
  });
});
