/* @vitest-environment jsdom */
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getCallbacksCall, getRouterSettingsCall } from "@/components/networking";
import ModelGroupAliasPanel from "./ModelGroupAliasPanel";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token", userId: "user-1", userRole: "Admin" }),
}));

vi.mock("@/components/networking", () => ({
  getCallbacksCall: vi.fn(),
  getRouterSettingsCall: vi.fn(),
  setCallbacksCall: vi.fn(),
}));

const aliases = {
  "fast-model": "claude-haiku",
  "smart-model": { model: "claude-sonnet", hidden: true },
};

describe("ModelGroupAliasPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getCallbacksCall).mockResolvedValue({ router_settings: { model_group_alias: aliases } });
    vi.mocked(getRouterSettingsCall).mockResolvedValue({ source: { model_group_alias: "db" } });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("locks delete controls when aliases are config-owned", async () => {
    vi.mocked(getRouterSettingsCall).mockResolvedValue({ source: { model_group_alias: "config" } });

    render(<ModelGroupAliasPanel />);

    expect(
      await screen.findByText(
        "These aliases are defined in config.yaml and are read-only here. Edit the config file to change them.",
      ),
    ).toBeInTheDocument();
    expect(await screen.findByText("fast-model")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Delete alias smart-model" })).toBeDisabled();
  });

  it("keeps aliases locked while ownership is loading", async () => {
    let resolveOwnership!: (value: Awaited<ReturnType<typeof getRouterSettingsCall>>) => void;
    const pendingOwnership = new Promise<Awaited<ReturnType<typeof getRouterSettingsCall>>>((resolve) => {
      resolveOwnership = resolve;
    });
    vi.mocked(getRouterSettingsCall).mockReturnValue(pendingOwnership);

    render(<ModelGroupAliasPanel />);

    expect(await screen.findByText("fast-model")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeDisabled();

    resolveOwnership({ source: { model_group_alias: "db" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeEnabled());
  });

  it("leaves delete controls enabled when aliases are database-managed", async () => {
    render(<ModelGroupAliasPanel />);

    await waitFor(() => expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeEnabled());
    expect(screen.getByRole("button", { name: "Delete alias smart-model" })).toBeEnabled();
  });

  it("renders aliases with enabled controls when router settings cannot be fetched", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(getRouterSettingsCall).mockRejectedValue(new Error("router settings unavailable"));

    render(<ModelGroupAliasPanel />);

    expect(await screen.findByText("fast-model")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeEnabled());
  });

  it("still shows the config-owned notice when callbacks cannot be fetched", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(getCallbacksCall).mockRejectedValue(new Error("callbacks unavailable"));
    vi.mocked(getRouterSettingsCall).mockResolvedValue({ source: { model_group_alias: "config" } });

    render(<ModelGroupAliasPanel />);

    expect(
      await screen.findByText(
        "These aliases are defined in config.yaml and are read-only here. Edit the config file to change them.",
      ),
    ).toBeInTheDocument();
  });
});
