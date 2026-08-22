import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/ModelSelect/ModelSelect", () => ({
  ModelSelect: ({ onChange }: { onChange: (values: string[]) => void }) => (
    <button type="button" onClick={() => onChange(["gpt-5.2"])}>
      set-models
    </button>
  ),
}));
vi.mock("@/app/(dashboard)/hooks/agents/useAgents", () => ({
  useAgents: () => ({ data: { agents: [{ agent_id: "agent-1", agent_name: "Support Agent" }] } }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [{ server_id: "srv-1", server_name: "GitHub MCP" }] }),
}));

import { AccessGroupCreateDialog } from "./AccessGroupCreateDialog";

const Harness = ({ createAccessGroup }: { createAccessGroup: (body: unknown) => Promise<unknown> }) => {
  const [open, setOpen] = React.useState(true);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        reopen
      </button>
      <AccessGroupCreateDialog open={open} onOpenChange={setOpen} createAccessGroup={createAccessGroup} />
    </>
  );
};

const renderDialog = (overrides?: { createAccessGroup?: ReturnType<typeof vi.fn> }) => {
  const createAccessGroup = overrides?.createAccessGroup ?? vi.fn().mockResolvedValue({});
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <Harness createAccessGroup={createAccessGroup} />
    </QueryClientProvider>,
  );
  return { createAccessGroup };
};

describe("AccessGroupCreateDialog", () => {
  it("blocks submit and shows an error when the name is missing", async () => {
    const user = userEvent.setup();
    const { createAccessGroup } = renderDialog();

    await user.click(screen.getByRole("button", { name: "Create Group" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Please enter the access group name");
    expect(createAccessGroup).not.toHaveBeenCalled();
  });

  it("returns to the General Info tab when submitting an invalid form from another tab", async () => {
    const user = userEvent.setup();
    const { createAccessGroup } = renderDialog();

    await user.click(screen.getByRole("tab", { name: "Models" }));
    await waitFor(() => expect(screen.queryByLabelText("Group Name")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Create Group" }));

    expect(await screen.findByLabelText("Group Name")).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent("Please enter the access group name");
    expect(createAccessGroup).not.toHaveBeenCalled();
  });

  it("sends only the group name for a minimal create and closes the dialog", async () => {
    const user = userEvent.setup();
    const { createAccessGroup } = renderDialog();

    await user.type(screen.getByLabelText("Group Name"), "prod-models");
    await user.click(screen.getByRole("button", { name: "Create Group" }));

    await waitFor(() => expect(createAccessGroup).toHaveBeenCalledTimes(1));
    expect(createAccessGroup.mock.calls[0][0]).toStrictEqual({ access_group_name: "prod-models" });
    await waitFor(() => expect(screen.queryByLabelText("Group Name")).not.toBeInTheDocument());
  });

  it("maps the description and model selections into the create body", async () => {
    const user = userEvent.setup();
    const { createAccessGroup } = renderDialog();

    await user.type(screen.getByLabelText("Group Name"), "prod-models");
    await user.type(screen.getByLabelText("Description"), "engineering access");
    await user.click(screen.getByRole("tab", { name: "Models" }));
    await user.click(screen.getByRole("button", { name: "set-models" }));
    await user.click(screen.getByRole("button", { name: "Create Group" }));

    await waitFor(() => expect(createAccessGroup).toHaveBeenCalledTimes(1));
    expect(createAccessGroup.mock.calls[0][0]).toStrictEqual({
      access_group_name: "prod-models",
      description: "engineering access",
      access_model_names: ["gpt-5.2"],
    });
  });

  it("keeps the dialog open with the entered values when the create fails", async () => {
    const user = userEvent.setup();
    const { createAccessGroup } = renderDialog({
      createAccessGroup: vi.fn().mockRejectedValue(new Error("boom")),
    });

    await user.type(screen.getByLabelText("Group Name"), "prod-models");
    await user.click(screen.getByRole("button", { name: "Create Group" }));

    await waitFor(() => expect(createAccessGroup).toHaveBeenCalledTimes(1));
    expect(screen.getByLabelText("Group Name")).toHaveValue("prod-models");
  });

  it("resets the form when the dialog is cancelled and reopened", async () => {
    const user = userEvent.setup();
    renderDialog();

    await user.type(screen.getByLabelText("Group Name"), "abandoned");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByLabelText("Group Name")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "reopen" }));
    expect(screen.getByLabelText("Group Name")).toHaveValue("");
  });

  it("resets the form when the dialog is dismissed with Escape and reopened", async () => {
    const user = userEvent.setup();
    renderDialog();

    await user.type(screen.getByLabelText("Group Name"), "abandoned");
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByLabelText("Group Name")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "reopen" }));
    expect(screen.getByLabelText("Group Name")).toHaveValue("");
  });

  it("cannot be dismissed while a create is pending, then closes once on success", async () => {
    const user = userEvent.setup();
    let resolveCreate: (value: unknown) => void = () => {};
    const createAccessGroup = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCreate = resolve;
        }),
    );
    renderDialog({ createAccessGroup });

    await user.type(screen.getByLabelText("Group Name"), "prod-models");
    await user.keyboard("{Enter}");
    await waitFor(() => expect(createAccessGroup).toHaveBeenCalledTimes(1));

    await user.keyboard("{Escape}");
    expect(screen.getByLabelText("Group Name")).toHaveValue("prod-models");

    resolveCreate({});
    await waitFor(() => expect(screen.queryByLabelText("Group Name")).not.toBeInTheDocument());
  });

  it("does not fire a second create while one is pending", async () => {
    const user = userEvent.setup();
    let resolveCreate: (value: unknown) => void = () => {};
    const createAccessGroup = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCreate = resolve;
        }),
    );
    renderDialog({ createAccessGroup });

    await user.type(screen.getByLabelText("Group Name"), "prod-models");
    await user.keyboard("{Enter}");
    await waitFor(() => expect(createAccessGroup).toHaveBeenCalledTimes(1));
    await user.keyboard("{Enter}");

    expect(createAccessGroup).toHaveBeenCalledTimes(1);
    resolveCreate({});
    await waitFor(() => expect(screen.queryByLabelText("Group Name")).not.toBeInTheDocument());
  });
});
