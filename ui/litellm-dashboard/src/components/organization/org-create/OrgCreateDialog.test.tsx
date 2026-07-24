import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));
vi.mock("@/components/ModelSelect/ModelSelect", () => ({
  ModelSelect: ({ onChange }: { onChange: (values: string[]) => void }) => (
    <button type="button" onClick={() => onChange(["gpt-5.2"])}>
      set-models
    </button>
  ),
}));
vi.mock("@/components/vector_store_management/VectorStoreSelector", () => ({
  __esModule: true,
  default: ({ onChange }: { onChange: (values: string[]) => void }) => (
    <button type="button" onClick={() => onChange(["vs-1"])}>
      set-vector-stores
    </button>
  ),
}));
vi.mock("@/components/mcp_server_management/MCPServerSelector", () => ({
  __esModule: true,
  default: ({
    onChange,
  }: {
    onChange: (values: { servers: string[]; accessGroups: string[]; toolsets: string[] }) => void;
  }) => (
    <button type="button" onClick={() => onChange({ servers: ["srv-1"], accessGroups: [], toolsets: ["ts-1"] })}>
      set-mcp
    </button>
  ),
}));

import { OrgCreateDialog } from "./OrgCreateDialog";

const Harness = ({ createOrganization }: { createOrganization: (body: unknown) => Promise<unknown> }) => {
  const [open, setOpen] = React.useState(true);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        reopen
      </button>
      <OrgCreateDialog open={open} onOpenChange={setOpen} accessToken="token" createOrganization={createOrganization} />
    </>
  );
};

const renderDialog = (overrides?: { createOrganization?: ReturnType<typeof vi.fn> }) => {
  const createOrganization = overrides?.createOrganization ?? vi.fn().mockResolvedValue({});
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <Harness createOrganization={createOrganization} />
    </QueryClientProvider>,
  );
  return { createOrganization };
};

describe("OrgCreateDialog", () => {
  it("blocks submit and shows an error when the name is missing", async () => {
    const user = userEvent.setup();
    const { createOrganization } = renderDialog();

    await user.click(screen.getByRole("button", { name: "Create Organization" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Please input an organization name");
    expect(createOrganization).not.toHaveBeenCalled();
  });

  it("sends only alias and models for a minimal create and closes the dialog", async () => {
    const user = userEvent.setup();
    const { createOrganization } = renderDialog();

    await user.type(screen.getByLabelText("Organization Name"), "new-org");
    await user.click(screen.getByRole("button", { name: "Create Organization" }));

    await waitFor(() => expect(createOrganization).toHaveBeenCalledTimes(1));
    expect(createOrganization.mock.calls[0][0]).toStrictEqual({ organization_alias: "new-org", models: [] });
    await waitFor(() => expect(screen.queryByLabelText("Organization Name")).not.toBeInTheDocument());
  });

  it("maps selectors and limits into the create body", async () => {
    const user = userEvent.setup();
    const { createOrganization } = renderDialog();

    await user.type(screen.getByLabelText("Organization Name"), "new-org");
    await user.click(screen.getByRole("button", { name: "set-models" }));
    await user.type(screen.getByLabelText("Tokens per minute Limit (TPM)"), "1000");
    await user.click(screen.getByRole("button", { name: "set-vector-stores" }));
    await user.click(screen.getByRole("button", { name: "set-mcp" }));
    await user.click(screen.getByRole("button", { name: "Create Organization" }));

    await waitFor(() => expect(createOrganization).toHaveBeenCalledTimes(1));
    const expectedBody = {
      organization_alias: "new-org",
      models: ["gpt-5.2"],
      tpm_limit: 1000,
      object_permission: {
        vector_stores: ["vs-1"],
        mcp_servers: ["srv-1"],
        mcp_toolsets: ["ts-1"],
      },
    };
    expect(createOrganization.mock.calls[0][0]).toStrictEqual(expectedBody);
  });

  it("blocks submit and shows an error for invalid metadata JSON", async () => {
    const user = userEvent.setup();
    const { createOrganization } = renderDialog();

    await user.type(screen.getByLabelText("Organization Name"), "new-org");
    await user.type(screen.getByLabelText("Metadata"), "not json");
    await user.click(screen.getByRole("button", { name: "Create Organization" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Metadata must be a valid JSON object");
    expect(createOrganization).not.toHaveBeenCalled();
  });

  it("keeps the dialog open with the entered values when the create fails", async () => {
    const user = userEvent.setup();
    const { createOrganization } = renderDialog({
      createOrganization: vi.fn().mockRejectedValue(new Error("boom")),
    });

    await user.type(screen.getByLabelText("Organization Name"), "new-org");
    await user.click(screen.getByRole("button", { name: "Create Organization" }));

    await waitFor(() => expect(createOrganization).toHaveBeenCalledTimes(1));
    expect(screen.getByLabelText("Organization Name")).toHaveValue("new-org");
  });

  it("resets the form when the dialog is cancelled and reopened", async () => {
    const user = userEvent.setup();
    renderDialog();

    await user.type(screen.getByLabelText("Organization Name"), "abandoned");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByLabelText("Organization Name")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "reopen" }));
    expect(screen.getByLabelText("Organization Name")).toHaveValue("");
  });
});
