import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { translate, type Language } from "@/i18n/language";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import { ApiError } from "@/lib/http/client";

vi.unmock("@/lib/toast");

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

const renderDialog = (overrides?: { createAccessGroup?: Mock; language?: Language }) => {
  const createAccessGroup = overrides?.createAccessGroup ?? vi.fn().mockResolvedValue({});
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <LanguageProvider
        defaultLanguage={overrides?.language ?? "en"}
        userId="access-create"
        storageScope="access-create"
      >
        <LanguageSelector />
        <Harness createAccessGroup={createAccessGroup} />
        <Toaster expand />
      </LanguageProvider>
    </QueryClientProvider>,
  );
  return { createAccessGroup };
};

describe("AccessGroupCreateDialog 表单集成", () => {
  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
  });

  afterEach(() => act(() => toast.dismiss()));

  it.each(["success", "failure", "server", "empty"])("访问组创建的 %s 通知切换语言且不重发请求", async (result) => {
    const user = userEvent.setup();
    const request = Promise.withResolvers<unknown>();
    const createAccessGroup = vi.fn().mockReturnValue(request.promise);
    renderDialog({ createAccessGroup });
    fireEvent.change(screen.getByLabelText("Group Name"), { target: { value: "Settings" } });
    await user.click(screen.getByRole("button", { name: "Create Group" }));
    await waitFor(() => expect(createAccessGroup).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    await act(async () => {
      if (result === "success") request.resolve({});
      else if (result === "server") request.reject(new ApiError("Home", 403, { error: { message: "Home" } }));
      else request.reject(result === "empty" ? new Error("") : null);
    });
    const message = result === "success" ? "Access group created successfully" : "Failed to create access group";
    expect(await screen.findByText(result === "server" ? "Home" : translate("zh-CN", message))).toBeVisible();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
    if (result === "server") expect(screen.getByText(translate("zh-CN", "Access Denied"))).toBeVisible();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(await screen.findByText(result === "server" ? "Home" : message)).toBeVisible();
    if (result === "server") expect(screen.getByText("Access Denied")).toBeVisible();
    expect(createAccessGroup).toHaveBeenCalledTimes(1);
    expect(createAccessGroup.mock.calls[0][0]).toStrictEqual({ access_group_name: "Settings" });
    if (result !== "success") expect(screen.getByLabelText("Group Name")).toHaveValue("Settings");
  });

  it.each<Language>(["en", "zh-CN"])("访问组校验随 %s 切换且保留跨标签页草稿", async (initialLanguage) => {
    const user = userEvent.setup();
    const nextLanguage = initialLanguage === "en" ? "zh-CN" : "en";
    const { createAccessGroup } = renderDialog({ language: initialLanguage });
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Group Name")), { target: { value: "  " } });
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Description")), { target: { value: "Home" } });
    await user.click(screen.getByRole("tab", { name: translate(initialLanguage, "Models"), exact: true }));
    await user.click(screen.getByRole("button", { name: "set-models" }));
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Create Group") }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      translate(initialLanguage, "Please enter the access group name"),
    );

    // 标签页和语言切换不能清空草稿或绕过必填校验。
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Interface language")), {
      target: { value: nextLanguage },
    });
    expect(screen.getByRole("alert")).toHaveTextContent(translate(nextLanguage, "Please enter the access group name"));
    expect(screen.getByLabelText(translate(nextLanguage, "Group Name"))).toHaveValue("  ");
    expect(screen.getByLabelText(translate(nextLanguage, "Description"))).toHaveValue("Home");
    expect(createAccessGroup).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText(translate(nextLanguage, "Group Name")), { target: { value: " Settings " } });
    await user.click(screen.getByRole("button", { name: translate(nextLanguage, "Create Group") }));
    await waitFor(() => expect(createAccessGroup).toHaveBeenCalledTimes(1));
    expect(createAccessGroup.mock.calls[0][0]).toStrictEqual({
      access_group_name: "Settings",
      description: "Home",
      access_model_names: ["gpt-5.2"],
    });
  });

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

  it("sends MCP servers and agents picked from the chip selectors as ids", async () => {
    const user = userEvent.setup();
    const { createAccessGroup } = renderDialog();

    await user.type(screen.getByLabelText("Group Name"), "mcp-group");
    await user.click(screen.getByRole("tab", { name: "MCP Servers" }));
    await user.click(screen.getByLabelText("Allowed MCP Servers"));
    await user.click(await screen.findByRole("option", { name: "GitHub MCP" }));
    expect(screen.getByLabelText("GitHub MCP")).toHaveAttribute("data-slot", "combobox-chip");
    await user.keyboard("{Escape}");

    await user.click(screen.getByRole("tab", { name: "Agents" }));
    await user.click(screen.getByLabelText("Allowed Agents"));
    await user.click(await screen.findByRole("option", { name: "Support Agent" }));
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Create Group" }));

    await waitFor(() => expect(createAccessGroup).toHaveBeenCalledTimes(1));
    expect(createAccessGroup.mock.calls[0][0]).toStrictEqual({
      access_group_name: "mcp-group",
      access_mcp_server_ids: ["srv-1"],
      access_agent_ids: ["agent-1"],
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
