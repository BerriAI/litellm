import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { accessGroupKeys, type AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));
vi.mock("@/components/ModelSelect/ModelSelect", () => ({
  ModelSelect: ({ value, onChange }: { value: string[]; onChange: (values: string[]) => void }) => (
    <>
      <span data-testid="models-value">{value.join(",")}</span>
      <button type="button" onClick={() => onChange(["gpt-5.2", "claude-sonnet-5"])}>
        set-models
      </button>
      <button type="button" onClick={() => onChange([])}>
        clear-models
      </button>
    </>
  ),
}));
vi.mock("@/app/(dashboard)/hooks/agents/useAgents", () => ({
  useAgents: () => ({ data: { agents: [{ agent_id: "agent-1", agent_name: "Support Agent" }] } }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [{ server_id: "srv-1", server_name: "GitHub MCP" }] }),
}));

import { AccessGroupEditDialog } from "./AccessGroupEditDialog";

const GROUP: AccessGroupResponse = {
  access_group_id: "ag-1",
  access_group_name: "prod-models",
  description: "Original description",
  access_model_names: ["gpt-5.2"],
  access_mcp_server_ids: [],
  access_agent_ids: [],
  assigned_team_ids: [],
  assigned_key_ids: [],
  created_at: "2026-01-01T00:00:00Z",
  created_by: "admin",
  updated_at: "2026-01-01T00:00:00Z",
  updated_by: "admin",
};

type PatchFn = (id: string, body: unknown) => Promise<AccessGroupResponse | undefined>;

const Harness = ({ patchAccessGroup, group }: { patchAccessGroup: PatchFn; group: AccessGroupResponse }) => {
  const [open, setOpen] = React.useState(true);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        reopen
      </button>
      <AccessGroupEditDialog
        open={open}
        onOpenChange={setOpen}
        accessGroup={group}
        patchAccessGroup={patchAccessGroup}
      />
    </>
  );
};

const renderDialog = (overrides?: { patchAccessGroup?: ReturnType<typeof vi.fn>; group?: AccessGroupResponse }) => {
  const patchAccessGroup = overrides?.patchAccessGroup ?? vi.fn().mockResolvedValue(undefined);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <Harness patchAccessGroup={patchAccessGroup} group={overrides?.group ?? GROUP} />
    </QueryClientProvider>,
  );
  return { patchAccessGroup, queryClient };
};

const saveButton = () => screen.getByRole("button", { name: /Save Changes|Saving\.\.\./ });

describe("AccessGroupEditDialog", () => {
  it("hydrates the form from the access group and disables Save until something changes", () => {
    renderDialog();

    expect(screen.getByLabelText("Group Name")).toHaveValue("prod-models");
    expect(screen.getByLabelText("Description")).toHaveValue("Original description");
    expect(saveButton()).toBeDisabled();
  });

  it("sends only the changed field and closes the dialog", async () => {
    const user = userEvent.setup();
    const { patchAccessGroup } = renderDialog();

    await user.clear(screen.getByLabelText("Description"));
    await user.type(screen.getByLabelText("Description"), "New description");
    await user.click(saveButton());

    await waitFor(() => expect(patchAccessGroup).toHaveBeenCalledTimes(1));
    expect(patchAccessGroup.mock.calls[0]).toStrictEqual(["ag-1", { description: "New description" }]);
    await waitFor(() => expect(screen.queryByLabelText("Group Name")).not.toBeInTheDocument());
  });

  it("sends every field edited across tabs before the first save, not just the first one", async () => {
    const user = userEvent.setup();
    const { patchAccessGroup } = renderDialog();

    await user.click(screen.getByRole("tab", { name: "MCP Servers" }));
    await user.click(screen.getByRole("combobox", { name: "Allowed MCP Servers" }));
    await user.click(await screen.findByRole("option", { name: "GitHub MCP" }));
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("tab", { name: "Agents" }));
    await user.click(screen.getByRole("combobox", { name: "Allowed Agents" }));
    await user.click(await screen.findByRole("option", { name: "Support Agent" }));
    await user.keyboard("{Escape}");
    await user.click(saveButton());

    await waitFor(() => expect(patchAccessGroup).toHaveBeenCalledTimes(1));
    expect(patchAccessGroup.mock.calls[0][1]).toStrictEqual({
      access_mcp_server_ids: ["srv-1"],
      access_agent_ids: ["agent-1"],
    });
  });

  it("clears the description with null when it is blanked out", async () => {
    const user = userEvent.setup();
    const { patchAccessGroup } = renderDialog();

    await user.clear(screen.getByLabelText("Description"));
    await user.click(saveButton());

    await waitFor(() => expect(patchAccessGroup).toHaveBeenCalledTimes(1));
    expect(patchAccessGroup.mock.calls[0][1]).toStrictEqual({ description: null });
  });

  it("does not send a field that was edited back to its original value", async () => {
    const user = userEvent.setup();
    const { patchAccessGroup } = renderDialog();

    await user.type(screen.getByLabelText("Group Name"), "x");
    await user.type(screen.getByLabelText("Group Name"), "{Backspace}");
    await user.clear(screen.getByLabelText("Description"));
    await user.type(screen.getByLabelText("Description"), "New description");
    await user.click(saveButton());

    await waitFor(() => expect(patchAccessGroup).toHaveBeenCalledTimes(1));
    expect(patchAccessGroup.mock.calls[0][1]).toStrictEqual({ description: "New description" });
  });

  it("sends an emptied model list as [] so the grant is removed", async () => {
    const user = userEvent.setup();
    const { patchAccessGroup } = renderDialog();

    await user.click(screen.getByRole("tab", { name: "Models" }));
    expect(screen.getByTestId("models-value")).toHaveTextContent("gpt-5.2");
    await user.click(screen.getByRole("button", { name: "clear-models" }));
    await user.click(saveButton());

    await waitFor(() => expect(patchAccessGroup).toHaveBeenCalledTimes(1));
    expect(patchAccessGroup.mock.calls[0][1]).toStrictEqual({ access_model_names: [] });
  });

  it("blocks submit, shows an error, and returns to General Info when the name is blanked", async () => {
    const user = userEvent.setup();
    const { patchAccessGroup } = renderDialog();

    await user.clear(screen.getByLabelText("Group Name"));
    await user.click(screen.getByRole("tab", { name: "Models" }));
    await waitFor(() => expect(screen.queryByLabelText("Group Name")).not.toBeInTheDocument());
    await user.click(saveButton());

    expect(await screen.findByLabelText("Group Name")).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent("Please enter the access group name");
    expect(patchAccessGroup).not.toHaveBeenCalled();
  });

  it("writes the returned record into the detail cache on success", async () => {
    const user = userEvent.setup();
    const updated: AccessGroupResponse = {
      ...GROUP,
      description: "New description",
      updated_at: "2026-02-01T00:00:00Z",
    };
    const { queryClient } = renderDialog({ patchAccessGroup: vi.fn().mockResolvedValue(updated) });

    await user.clear(screen.getByLabelText("Description"));
    await user.type(screen.getByLabelText("Description"), "New description");
    await user.click(saveButton());

    await waitFor(() => expect(queryClient.getQueryData(accessGroupKeys.detail("ag-1"))).toStrictEqual(updated));
  });

  it("keeps the dialog open with the edited values when the save fails", async () => {
    const user = userEvent.setup();
    const { patchAccessGroup } = renderDialog({ patchAccessGroup: vi.fn().mockRejectedValue(new Error("boom")) });

    await user.clear(screen.getByLabelText("Description"));
    await user.type(screen.getByLabelText("Description"), "New description");
    await user.click(saveButton());

    await waitFor(() => expect(patchAccessGroup).toHaveBeenCalledTimes(1));
    expect(screen.getByLabelText("Description")).toHaveValue("New description");
  });

  it("discards edits when cancelled and reopened", async () => {
    const user = userEvent.setup();
    renderDialog();

    await user.clear(screen.getByLabelText("Description"));
    await user.type(screen.getByLabelText("Description"), "abandoned");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByLabelText("Description")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "reopen" }));
    expect(screen.getByLabelText("Description")).toHaveValue("Original description");
    expect(saveButton()).toBeDisabled();
  });

  it("discards edits when dismissed with Escape and reopened", async () => {
    const user = userEvent.setup();
    renderDialog();

    await user.clear(screen.getByLabelText("Description"));
    await user.type(screen.getByLabelText("Description"), "abandoned");
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByLabelText("Description")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "reopen" }));
    expect(screen.getByLabelText("Description")).toHaveValue("Original description");
  });

  it("cannot be dismissed while a save is pending, then closes once on success", async () => {
    const user = userEvent.setup();
    let resolveSave: (value: AccessGroupResponse | undefined) => void = () => {};
    const patchAccessGroup = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveSave = resolve;
        }),
    );
    renderDialog({ patchAccessGroup });

    await user.clear(screen.getByLabelText("Description"));
    await user.type(screen.getByLabelText("Description"), "New description");
    await user.type(screen.getByLabelText("Group Name"), "{Enter}");
    await waitFor(() => expect(patchAccessGroup).toHaveBeenCalledTimes(1));

    await user.keyboard("{Escape}");
    expect(screen.getByLabelText("Description")).toHaveValue("New description");

    await user.type(screen.getByLabelText("Group Name"), "{Enter}");
    expect(patchAccessGroup).toHaveBeenCalledTimes(1);

    resolveSave(undefined);
    await waitFor(() => expect(screen.queryByLabelText("Description")).not.toBeInTheDocument());
  });
});
