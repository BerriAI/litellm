import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MCPServer, MCPSubmissionsSummary } from "@/components/mcp_tools/types";
import { fetchMCPSubmissions, getGeneralSettingsCall, updateConfigFieldSetting } from "@/components/networking";
import { MCPSubmissionsTab } from "./MCPSubmissionsTab";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { act, fireEvent, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  approveMCPServer: vi.fn(),
  fetchMCPSubmissions: vi.fn(),
  getGeneralSettingsCall: vi.fn(),
  rejectMCPServer: vi.fn(),
  updateConfigFieldSetting: vi.fn(),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn() },
}));

const token = "token";

const pendingSubmission: MCPServer = {
  server_id: "submission-1",
  server_name: "submission",
  alias: "Submission server",
  description: "A server with required details",
  url: "https://example.com/mcp",
  transport: "http",
  auth_type: "none",
  approval_status: "pending_review",
  created_at: "2026-01-01T00:00:00Z",
  created_by: "user",
  updated_at: "2026-01-01T00:00:00Z",
  updated_by: "user",
};

const submissionSummary: MCPSubmissionsSummary = {
  total: 1,
  pending_review: 1,
  active: 0,
  rejected: 0,
  items: [pendingSubmission],
};

const configuredRules = [
  {
    field_name: "mcp_required_fields",
    field_type: "List",
    field_value: ["description", "url", "alias"],
    field_description: "Required MCP submission fields",
    stored_in_db: true,
  },
];

describe("MCPSubmissionsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchMCPSubmissions).mockResolvedValue(submissionSummary);
    vi.mocked(getGeneralSettingsCall).mockResolvedValue(configuredRules);
    vi.mocked(updateConfigFieldSetting).mockResolvedValue(undefined);
  });

  it("renders configured submission rules and card compliance checks", async () => {
    renderWithProviders(<MCPSubmissionsTab accessToken={token} />);

    expect(await screen.findByText("(3 required fields)")).toBeInTheDocument();
    expect(screen.queryByText("no rules set")).not.toBeInTheDocument();
    expect(screen.getByText("3 passing, 0 failing")).toBeInTheDocument();
    expect(screen.getAllByText("Passes")).toHaveLength(3);
  });

  it("preloads the rules editor and preserves them when adding a field", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MCPSubmissionsTab accessToken={token} />);

    await user.click(await screen.findByText("Submission Rules"));

    expect(screen.getByRole("checkbox", { name: /Description/ })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /Server URL/ })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /Alias/ })).toBeChecked();
    const authTypeCheckbox = screen.getByRole("checkbox", { name: /Auth configured/ });
    expect(authTypeCheckbox).not.toBeChecked();

    await user.click(authTypeCheckbox);
    await user.click(screen.getByRole("button", { name: "Save Rules" }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("token", "mcp_required_fields", [
        "description",
        "url",
        "alias",
        "auth_type",
      ]),
    );
  });

  it("keeps the rules editor disabled until saved rules load", async () => {
    const user = userEvent.setup();
    const { promise: settingsPromise, resolve: resolveSettings } =
      Promise.withResolvers<Awaited<ReturnType<typeof getGeneralSettingsCall>>>();
    vi.mocked(getGeneralSettingsCall).mockReturnValue(settingsPromise);

    renderWithProviders(<MCPSubmissionsTab accessToken={token} />);
    await user.click(screen.getByText("Submission Rules"));

    const descriptionCheckbox = screen.getByRole("checkbox", { name: /Description/ });
    const serverUrlCheckbox = screen.getByRole("checkbox", { name: /Server URL/ });
    const aliasCheckbox = screen.getByRole("checkbox", { name: /Alias/ });
    const authTypeCheckbox = screen.getByRole("checkbox", { name: /Auth configured/ });
    const saveRulesButton = screen.getByRole("button", { name: "Save Rules" });

    expect(descriptionCheckbox).toBeDisabled();
    expect(serverUrlCheckbox).toBeDisabled();
    expect(aliasCheckbox).toBeDisabled();
    expect(authTypeCheckbox).toBeDisabled();
    expect(saveRulesButton).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();

    await act(async () => {
      resolveSettings(configuredRules);
    });

    expect(descriptionCheckbox).toBeEnabled();
    expect(serverUrlCheckbox).toBeEnabled();
    expect(aliasCheckbox).toBeEnabled();
    expect(authTypeCheckbox).toBeEnabled();
    expect(descriptionCheckbox).toBeChecked();
    expect(serverUrlCheckbox).toBeChecked();
    expect(aliasCheckbox).toBeChecked();
    expect(authTypeCheckbox).not.toBeChecked();
    expect(saveRulesButton).toBeEnabled();
  });

  it("shows no rules when the general settings array is empty", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([]);

    renderWithProviders(<MCPSubmissionsTab accessToken={token} />);

    expect(await screen.findByText("no rules set")).toBeInTheDocument();
  });
});

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

const renderAt = (searchParams: string, onUrlUpdate?: OnUrlUpdateFunction) =>
  renderWithProviders(<MCPSubmissionsTab accessToken={token} />, { searchParams, onUrlUpdate });

const filterSummary: MCPSubmissionsSummary = {
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
    vi.clearAllMocks();
    vi.mocked(fetchMCPSubmissions).mockResolvedValue(filterSummary);
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([]);
  });

  it("lists every submission without filters", async () => {
    renderAt("");

    await screen.findByRole("heading", { level: 3, name: "linear-pending" });
    expect(shownSubmissions()).toEqual(SUBMISSION_NAMES);
  });

  it("applies the search and status in the URL", async () => {
    renderAt("?sub_search=github&sub_status=pending_review");

    await screen.findByRole("heading", { level: 3, name: "github-pending" });
    expect(shownSubmissions()).toEqual(["github-pending"]);
    expect(screen.getByPlaceholderText("Search MCP servers...")).toHaveValue("github");
    expect(screen.getByRole("combobox")).toHaveValue("pending_review");
  });

  it("ignores an unknown status in the URL", async () => {
    renderAt("?sub_status=archived");

    await screen.findByRole("heading", { level: 3, name: "linear-pending" });
    expect(shownSubmissions()).toEqual(SUBMISSION_NAMES);
    expect(screen.getByRole("combobox")).toHaveValue("all");
  });

  it("writes the search and status the user picks", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderAt("?tab=submitted", onUrlUpdate);
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
