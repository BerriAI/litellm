import { describe, it, expect, vi, beforeEach } from "vitest";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { MCPSubmissionsTab } from "./MCPSubmissionsTab";
import { fetchMCPSubmissions } from "@/components/networking";
import type { MCPServer, MCPSubmissionsSummary } from "@/components/mcp_tools/types";
import { fireEvent, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  fetchMCPSubmissions: vi.fn(),
  approveMCPServer: vi.fn(),
  rejectMCPServer: vi.fn(),
  getGeneralSettingsCall: vi.fn().mockResolvedValue({ data: [] }),
  updateConfigFieldSetting: vi.fn(),
}));

const submission = (alias: string, approval_status: MCPServer["approval_status"]): MCPServer => ({
  server_id: `id-${alias}`,
  alias,
  url: `https://${alias}.example.com/mcp`,
  approval_status,
  created_at: "2024-01-01T00:00:00Z",
  created_by: "user-1",
  updated_at: "2024-01-01T00:00:00Z",
  updated_by: "user-1",
});

const SUBMISSION_NAMES = ["github-pending", "github-live", "linear-pending"];

const shownSubmissions = () =>
  screen
    .getAllByRole("heading", { level: 3 })
    .map((heading) => heading.textContent)
    .filter((name) => name !== null && SUBMISSION_NAMES.includes(name));

const renderTab = (searchParams: string, onUrlUpdate?: OnUrlUpdateFunction) =>
  renderWithProviders(<MCPSubmissionsTab accessToken="sk-test" />, { searchParams, onUrlUpdate });

const SUMMARY: MCPSubmissionsSummary = {
  total: 3,
  pending_review: 2,
  active: 1,
  rejected: 0,
  items: [
    submission("github-pending", "pending_review"),
    submission("github-live", "active"),
    submission("linear-pending", "pending_review"),
  ],
};

describe("MCPSubmissionsTab filter URL state", () => {
  beforeEach(() => {
    vi.mocked(fetchMCPSubmissions).mockResolvedValue(SUMMARY);
  });

  it("lists every submission without filters", async () => {
    renderTab("");

    await screen.findByRole("heading", { level: 3, name: "linear-pending" });
    expect(shownSubmissions()).toEqual(SUBMISSION_NAMES);
  });

  it("applies the search and status in the URL", async () => {
    renderTab("?sub_search=github&sub_status=pending_review");

    await screen.findByRole("heading", { level: 3, name: "github-pending" });
    expect(shownSubmissions()).toEqual(["github-pending"]);
    expect(screen.getByPlaceholderText("Search MCP servers...")).toHaveValue("github");
    expect(screen.getByRole("combobox")).toHaveValue("pending_review");
  });

  it("ignores an unknown status in the URL", async () => {
    renderTab("?sub_status=archived");

    await screen.findByRole("heading", { level: 3, name: "linear-pending" });
    expect(shownSubmissions()).toEqual(SUBMISSION_NAMES);
    expect(screen.getByRole("combobox")).toHaveValue("all");
  });

  it("writes the search and status the user picks", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderTab("?tab=submitted", onUrlUpdate);
    await screen.findByRole("heading", { level: 3, name: "linear-pending" });

    fireEvent.change(screen.getByPlaceholderText("Search MCP servers..."), { target: { value: "pending" } });

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("sub_search")).toBe("pending"));
    expect(shownSubmissions()).toEqual(["github-pending", "linear-pending"]);

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "active" } });

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("sub_status")).toBe("active"));
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("sub_search")).toBe("pending");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("tab")).toBe("submitted");
    expect(screen.getByText("No MCP server submissions match your filters.")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "all" } });

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("sub_status")).toBe(false));
    expect(shownSubmissions()).toEqual(["github-pending", "linear-pending"]);
  });
});
