import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const mocks = vi.hoisted(() => ({
  fetchMCPEventTypes: vi.fn(),
  servers: [] as { server_id: string; server_name?: string; events?: { enabled: boolean } }[],
}));
vi.mock("@/components/networking", () => ({ fetchMCPEventTypes: mocks.fetchMCPEventTypes }));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: mocks.servers }),
}));

import { MCPEventPermissionsEditor } from "./MCPEventPermissionsEditor";

const BUILD_FAILED = { name: "build.failed", description: "A CI build failed", delivery: ["poll"] };
const DEPLOY_DONE = { name: "deploy.done", description: "A deploy finished", delivery: ["poll"] };

function renderEditor(value: Record<string, string[]>, onSave = vi.fn().mockResolvedValue(undefined)) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MCPEventPermissionsEditor accessToken="sk-admin" value={value} onSave={onSave} />
    </QueryClientProvider>,
  );
  return onSave;
}

describe("MCPEventPermissionsEditor", () => {
  beforeEach(() => {
    mocks.servers = [
      { server_id: "srv-ci", server_name: "CI", events: { enabled: true } },
      { server_id: "srv-off", server_name: "Off", events: { enabled: false } },
    ];
    mocks.fetchMCPEventTypes.mockResolvedValue([BUILD_FAILED, DEPLOY_DONE]);
  });

  it("lists events-enabled servers by name and asks only them for event types", async () => {
    renderEditor({});
    expect(await screen.findByText("CI")).toBeInTheDocument();
    expect(screen.queryByText("Off")).not.toBeInTheDocument();
    expect(mocks.fetchMCPEventTypes).toHaveBeenCalledWith("sk-admin", "srv-ci");
    expect(mocks.fetchMCPEventTypes).not.toHaveBeenCalledWith("sk-admin", "srv-off");
  });

  it("saves a new restriction and keeps the restrictions on other servers", async () => {
    const user = userEvent.setup();
    const onSave = renderEditor({ "srv-gone": ["x"] });
    await screen.findByText("CI");
    const save = screen.getByRole("button", { name: "Save event permissions" });
    expect(save).toBeDisabled();
    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click(await screen.findByText("deploy.done"));
    await user.click(save);
    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ "srv-gone": ["x"], "srv-ci": ["deploy.done"] }));
  });

  it("shows a stored empty list as no event types, not all of them", async () => {
    renderEditor({ "srv-ci": [] });
    expect(await screen.findByPlaceholderText("No event types")).toBeInTheDocument();
  });

  it("offers a retry when event types fail to load", async () => {
    const user = userEvent.setup();
    mocks.fetchMCPEventTypes.mockRejectedValueOnce(new Error("offline"));
    renderEditor({});
    await user.click(await screen.findByRole("button", { name: "Retry" }));
    expect(await screen.findByText("CI")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps a stored restriction editable when its server lists no event types", async () => {
    mocks.fetchMCPEventTypes.mockResolvedValue([]);
    renderEditor({ "srv-ci": ["build.failed"] });
    expect(await screen.findByText("CI")).toBeInTheDocument();
    expect(screen.getByText("build.failed")).toBeInTheDocument();
  });

  it("keeps a restriction on a server that no longer publishes events so it can be cleared", async () => {
    const user = userEvent.setup();
    const onSave = renderEditor({ "srv-off": ["x"] });
    await screen.findByText("CI");
    expect(screen.getByText("srv-off")).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "Block all" })[1]);
    await user.click(screen.getByRole("button", { name: "Save event permissions" }));
    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ "srv-off": [] }));
  });

  it("blocks every event type on a server explicitly and allows them again", async () => {
    const user = userEvent.setup();
    const onSave = renderEditor({});
    await screen.findByText("CI");
    await user.click(screen.getByRole("button", { name: "Block all" }));
    expect(screen.getByPlaceholderText("No event types")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save event permissions" }));
    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ "srv-ci": [] }));
    await user.click(screen.getByRole("button", { name: "Allow all" }));
    expect(screen.getByPlaceholderText("All event types")).toBeInTheDocument();
  });

  it("renders nothing when no server publishes events", async () => {
    mocks.servers = [{ server_id: "srv-off", server_name: "Off", events: { enabled: false } }];
    renderEditor({});
    await waitFor(() => expect(screen.queryByText("Event permissions")).not.toBeInTheDocument());
  });
});
