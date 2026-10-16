import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "@/components/networking";

import TernaryCostTracking from "./TernaryCostTracking";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: () => ({ accessToken: "test-token" }),
}));

vi.mock("@/components/networking", () => ({
  apiClient: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

const mockGet = vi.mocked(apiClient.get);
const mockPost = vi.mocked(apiClient.post);
const mockPut = vi.mocked(apiClient.put);

const EMPTY_VIEW = {
  api_key_masked: null,
  connection_id: null,
  ternary_base_url: null,
  frequency: null,
  interval_seconds: null,
  status: null,
};

const CONFIGURED_VIEW = {
  api_key_masked: "te****ey",
  connection_id: "conn-123",
  ternary_base_url: "https://api.ternary.app",
  frequency: "daily",
  interval_seconds: null,
  status: "configured",
};

const SAVED = { message: "Ternary settings saved. Restart the proxy to apply the change", status: "success" };

const renderTab = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <TernaryCostTracking />
    </QueryClientProvider>,
  );
};

describe("TernaryCostTracking", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates an integration from the empty state and tells the admin to restart", async () => {
    mockGet.mockResolvedValueOnce(EMPTY_VIEW).mockResolvedValue(CONFIGURED_VIEW);
    mockPost.mockResolvedValue(SAVED);
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByRole("button", { name: "Add Ternary Integration" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("API Key"), { target: { value: "tern-key" } });
    fireEvent.change(within(dialog).getByLabelText("Connection ID"), { target: { value: "conn-123" } });
    await user.click(within(dialog).getByRole("button", { name: "Create" }));

    expect(await screen.findByText(SAVED.message)).toBeInTheDocument();
    expect(mockPost).toHaveBeenCalledWith("/ternary/init", {
      accessToken: "test-token",
      body: {
        api_key: "tern-key",
        connection_id: "conn-123",
        ternary_base_url: "https://api.ternary.app",
        frequency: "daily",
        interval_seconds: null,
      },
    });
    expect(await screen.findByText("te****ey")).toBeInTheDocument();
  });

  it("asks for an interval before saving an interval export", async () => {
    mockGet.mockResolvedValue(EMPTY_VIEW);
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByRole("button", { name: "Add Ternary Integration" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("API Key"), { target: { value: "tern-key" } });
    fireEvent.change(within(dialog).getByLabelText("Connection ID"), { target: { value: "conn-123" } });
    await user.click(within(dialog).getByRole("combobox", { name: "Export Frequency" }));
    await user.click(await screen.findByRole("option", { name: "Fixed interval (setup testing)" }));
    await user.click(within(dialog).getByRole("button", { name: "Create" }));

    expect(await within(dialog).findByText("Enter how often to export, in seconds")).toBeInTheDocument();
    expect(mockPost).not.toHaveBeenCalled();
  });

  it("keeps the stored API key when editing without typing a new one", async () => {
    mockGet.mockResolvedValue(CONFIGURED_VIEW);
    mockPut.mockResolvedValue(SAVED);
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByRole("button", { name: "Edit" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByLabelText("Connection ID")).toHaveValue("conn-123");
    fireEvent.change(within(dialog).getByLabelText("Connection ID"), { target: { value: "conn-456" } });
    await user.click(within(dialog).getByRole("button", { name: "Update" }));

    expect(await screen.findByText(SAVED.message)).toBeInTheDocument();
    expect(mockPut).toHaveBeenCalledWith("/ternary/settings", {
      accessToken: "test-token",
      body: {
        connection_id: "conn-456",
        ternary_base_url: "https://api.ternary.app",
        frequency: "daily",
        interval_seconds: null,
      },
    });
  });

  it("previews the export as an estimate without sending anything", async () => {
    mockGet.mockResolvedValue(CONFIGURED_VIEW);
    mockPost.mockResolvedValue({
      message: "Preview only",
      status: "success",
      usage_data: [],
      normalized_data: [{ Tags: '{"prompt_tokens": "10"}' }],
      summary: {
        total_records: 3,
        estimated_spend: 7.75,
        total_tokens: 66,
        unique_teams: 2,
        unique_models: 2,
        days: ["2026-10-13", "2026-10-14"],
      },
    });
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByRole("button", { name: "Preview Export" }));

    expect(await screen.findByText("Preview (LiteLLM estimated spend)")).toBeInTheDocument();
    expect(screen.getByText(/3 FOCUS rows across 2 day\(s\), estimated spend \$7\.75, 66 tokens/)).toBeInTheDocument();
    expect(mockPost).toHaveBeenCalledWith("/ternary/dry-run", { accessToken: "test-token", body: { limit: 500 } });
  });

  it("exports the scheduled window after confirmation and lists the days sent", async () => {
    mockGet.mockResolvedValue(CONFIGURED_VIEW);
    mockPost.mockResolvedValue({
      message: "Ternary export uploaded 2 day(s)",
      status: "success",
      start_time_utc: "2026-10-13T00:00:00Z",
      end_time_utc: "2026-10-15T22:00:00Z",
      days_uploaded: ["2026-10-14", "2026-10-15"],
    });
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByRole("button", { name: "Export Now" }));
    await user.click(await screen.findByRole("button", { name: "Export" }));

    expect(await screen.findByText("Export complete")).toBeInTheDocument();
    expect(screen.getByText(/Uploaded 2 day\(s\), one file each: 2026-10-14, 2026-10-15/)).toBeInTheDocument();
    expect(mockPost).toHaveBeenCalledWith("/ternary/export", { accessToken: "test-token", body: {} });
  });

  it("says nothing was sent when the window had no usage", async () => {
    mockGet.mockResolvedValue(CONFIGURED_VIEW);
    mockPost.mockResolvedValue({
      message: "No usage to upload in this window",
      status: "success",
      start_time_utc: "2026-10-13T00:00:00Z",
      end_time_utc: "2026-10-15T22:00:00Z",
      days_uploaded: [],
    });
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByRole("button", { name: "Export Now" }));
    await user.click(await screen.findByRole("button", { name: "Export" }));

    expect(await screen.findByText("Nothing to export")).toBeInTheDocument();
    expect(screen.getByText(/No usage between 2026-10-13T00:00:00Z and 2026-10-15T22:00:00Z/)).toBeInTheDocument();
    expect(screen.queryByText("Export complete")).not.toBeInTheDocument();
  });

  it("shows a config-file setup read-only but still lets the admin preview and export", async () => {
    mockGet.mockResolvedValue({ ...EMPTY_VIEW, status: "config_file" });
    renderTab();

    expect(await screen.findByText(/sets up Ternary in its config file/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add Ternary Integration" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export Now" })).toBeEnabled();
  });

  it("asks for the API key again before moving the integration to a new base URL", async () => {
    mockGet.mockResolvedValue(CONFIGURED_VIEW);
    const user = userEvent.setup();
    renderTab();

    await user.click(await screen.findByRole("button", { name: "Edit" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("Base URL"), { target: { value: "https://collector.example.net" } });
    await user.click(within(dialog).getByRole("button", { name: "Update" }));

    expect(await within(dialog).findByText("Enter the API key again when changing the base URL")).toBeInTheDocument();
    expect(mockPut).not.toHaveBeenCalled();
  });
});
