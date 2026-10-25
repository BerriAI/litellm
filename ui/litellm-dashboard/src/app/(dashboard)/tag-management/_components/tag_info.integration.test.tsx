import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { tagInfoCall, tagUpdateCall } from "@/components/networking";
import type { Tag } from "@/components/tag_management/types";

import TagInfoView from "./tag_info";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { translate, type Language } from "@/i18n/language";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import { ApiError } from "@/lib/http/client";

vi.unmock("@/lib/toast");

vi.mock("@/components/networking", () => ({
  tagInfoCall: vi.fn(),
  tagUpdateCall: vi.fn(),
}));

vi.mock("@/components/organisms/create_key_button", () => ({
  fetchUserModels: vi.fn(
    (_userID: string, _userRole: string, _accessToken: string, setUserModels: (models: string[]) => void) => {
      setUserModels(["model-1", "model-2"]);
      return Promise.resolve();
    },
  ),
}));

const mockTagInfoCall = vi.mocked(tagInfoCall);
const mockTagUpdateCall = vi.mocked(tagUpdateCall);

const tag: Tag = {
  name: "prod-tag",
  description: "original description",
  models: ["model-1", "model-2"],
  model_info: { "model-1": "GPT-4", "model-2": "Claude-3" },
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-02T00:00:00Z",
  litellm_budget_table: { max_budget: 10, budget_duration: "7d", tpm_limit: 1000, rpm_limit: 60 },
};

const renderEditor = async () => {
  const user = userEvent.setup();
  render(<TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag />);
  const nameInput = await screen.findByLabelText("Tag Name");
  return { user, nameInput };
};

describe("TagInfoView save payload", () => {
  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
    vi.clearAllMocks();
    mockTagInfoCall.mockResolvedValue({ "prod-tag": tag });
    mockTagUpdateCall.mockResolvedValue(undefined);
  });

  afterEach(() => act(() => toast.dismiss()));

  it.each(["load", "save"])("标签 %s 错误包装只显示业务消息并保留权限分类", async (operation) => {
    const user = userEvent.setup();
    const body = { error: { message: "Home", internal: "internal implementation detail" } };
    const error = new ApiError(JSON.stringify(body), 403, body);
    if (operation === "load") mockTagInfoCall.mockRejectedValue(error);
    else mockTagUpdateCall.mockRejectedValue(error);
    render(
      <LanguageProvider defaultLanguage="en" userId="tag-error-envelope" storageScope="tag-error-envelope">
        <LanguageSelector />
        <TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag />
        <Toaster expand />
      </LanguageProvider>,
    );
    if (operation === "save") {
      fireEvent.change(await screen.findByLabelText("Description"), { target: { value: "Settings" } });
      await user.click(screen.getByRole("button", { name: "Save Changes" }));
    }
    const message = operation === "load" ? "Error fetching tag details: {error}" : "Error updating tag: {error}";
    expect(await screen.findByText(translate("en", message, { error: "Home" }))).toBeVisible();
    expect(screen.getByText("Access Denied")).toBeVisible();
    expect(screen.queryByText(/internal implementation detail/)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(await screen.findByText(translate("zh-CN", message, { error: "Home" }))).toBeVisible();
    expect(screen.getByText(translate("zh-CN", "Access Denied"))).toBeVisible();
    expect(screen.queryByText(/internal implementation detail/)).not.toBeInTheDocument();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
    expect(mockTagInfoCall).toHaveBeenCalledTimes(1);
    expect(mockTagUpdateCall).toHaveBeenCalledTimes(operation === "save" ? 1 : 0);
    if (operation === "save") expect(screen.getByLabelText(translate("zh-CN", "Description"))).toHaveValue("Settings");
  });

  it("标签加载失败通知随语言切换，服务端同名错误保持原文", async () => {
    mockTagInfoCall.mockRejectedValue(new Error("Home"));
    render(
      <LanguageProvider defaultLanguage="en" userId="tag-load-notice" storageScope="tag-load-notice">
        <LanguageSelector />
        <TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag />
        <Toaster expand />
      </LanguageProvider>,
    );
    expect(await screen.findByText("Error fetching tag details: Home")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(
      await screen.findByText(translate("zh-CN", "Error fetching tag details: {error}", { error: "Home" })),
    ).toBeVisible();
    expect(screen.queryByText("Error fetching tag details: Home")).not.toBeInTheDocument();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
    expect(mockTagInfoCall).toHaveBeenCalledTimes(1);
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
  });

  it.each(["success", "failure"])("标签保存的 %s 通知跟随当前语言且保留服务端详情", async (result) => {
    const user = userEvent.setup();
    const request = Promise.withResolvers<void>();
    mockTagUpdateCall.mockReturnValue(request.promise);
    render(
      <LanguageProvider defaultLanguage="en" userId="tag-notice" storageScope="tag-notice">
        <LanguageSelector />
        <TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag />
        <Toaster expand />
      </LanguageProvider>,
    );
    fireEvent.change(await screen.findByLabelText("Description"), { target: { value: "Settings" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    await act(async () => {
      if (result === "success") request.resolve();
      else request.reject(new Error("Home"));
    });
    const message = result === "success" ? "Tag updated successfully" : "Error updating tag: {error}";
    const values = { error: "Home" };
    expect(await screen.findByText(translate("zh-CN", message, values))).toBeVisible();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(await screen.findByText(translate("en", message, values))).toBeVisible();
    expect(screen.queryByText(translate("zh-CN", message, values))).not.toBeInTheDocument();
    expect(mockTagUpdateCall).toHaveBeenCalledTimes(1);
    expect(mockTagInfoCall).toHaveBeenCalledTimes(result === "success" ? 2 : 1);
    if (result === "failure") expect(screen.getByLabelText("Description")).toHaveValue("Settings");
  });

  it.each<Language>(["en", "zh-CN"])("标签名称与预算的已有错误随 %s 切换且保留草稿", async (initialLanguage) => {
    const user = userEvent.setup();
    const nextLanguage = initialLanguage === "en" ? "zh-CN" : "en";
    render(
      <LanguageProvider defaultLanguage={initialLanguage} userId="tag-error-switch" storageScope="tag-error-switch">
        <LanguageSelector />
        <TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag />
      </LanguageProvider>,
    );
    fireEvent.change(await screen.findByLabelText(translate(initialLanguage, "Tag Name")), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Description")), {
      target: { value: "Settings" },
    });
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Budget & Rate Limits") }));
    fireEvent.change(await screen.findByLabelText(translate(initialLanguage, "Max Budget (USD)")), {
      target: { value: "-1" },
    });
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Save Changes") }));
    expect(await screen.findByText(translate(initialLanguage, "Please input a tag name"))).toBeVisible();
    expect(screen.getByText(translate(initialLanguage, "Enter a nonnegative budget"))).toBeVisible();
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Interface language")), {
      target: { value: nextLanguage },
    });
    expect(screen.getByText(translate(nextLanguage, "Please input a tag name"))).toBeVisible();
    expect(screen.getByText(translate(nextLanguage, "Enter a nonnegative budget"))).toBeVisible();
    expect(screen.getByLabelText(translate(nextLanguage, "Description"))).toHaveValue("Settings");
    expect(screen.getByLabelText(translate(nextLanguage, "Max Budget (USD)"))).toHaveValue(-1);
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
    expect(mockTagInfoCall).toHaveBeenCalledTimes(1);
    fireEvent.change(screen.getByLabelText(translate(nextLanguage, "Tag Name")), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText(translate(nextLanguage, "Max Budget (USD)")), { target: { value: "20.5" } });
    await user.click(screen.getByRole("button", { name: translate(nextLanguage, "Save Changes") }));
    const expectedPayload = {
      name: "Home",
      description: "Settings",
      max_budget: 20.5,
      models: ["model-1", "model-2"],
      budget_duration: "7d",
    };
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining(expectedPayload));
  });

  it("切换语言保留标签草稿，显示中文名称校验且不重新加载", async () => {
    localStorage.clear();
    render(
      <LanguageProvider defaultLanguage="en" userId="tag-validation" storageScope="test">
        <LanguageSelector />
        <TagInfoView tagId="prod-tag" onClose={vi.fn()} accessToken="sk-test" is_admin editTag />
      </LanguageProvider>,
    );
    fireEvent.change(await screen.findByLabelText("Description"), { target: { value: "Settings" } });
    fireEvent.change(screen.getByLabelText("Tag Name"), { target: { value: "" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    fireEvent.click(screen.getByRole("button", { name: translate("zh-CN", "Save Changes") }));
    expect(await screen.findByText("请输入标签名称")).toBeVisible();
    expect(screen.getByLabelText(translate("zh-CN", "Description"))).toHaveValue("Settings");
    expect(mockTagInfoCall).toHaveBeenCalledTimes(1);
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
  });

  it("leaves the budget unchanged while the budget section is collapsed", async () => {
    const { user, nameInput } = await renderEditor();

    await user.clear(nameInput);
    fireEvent.change(nameInput, { target: { value: "renamed-tag" } });

    const descriptionInput = screen.getByLabelText("Description");
    await user.clear(descriptionInput);
    fireEvent.change(descriptionInput, { target: { value: "updated description" } });

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    const expected = {
      name: "renamed-tag",
      description: "updated description",
      models: ["model-1", "model-2"],
      max_budget: undefined,
      tpm_limit: undefined,
      rpm_limit: undefined,
      budget_duration: undefined,
    };

    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expected);
  });

  it("should send the budget fields once the budget section is expanded", async () => {
    const { user, nameInput } = await renderEditor();
    expect(nameInput).toHaveValue("prod-tag");

    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));

    const maxBudgetInput = await screen.findByLabelText("Max Budget (USD)");
    await user.clear(maxBudgetInput);
    fireEvent.change(maxBudgetInput, { target: { value: "150.75" } });

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    const expected = {
      name: "prod-tag",
      description: "original description",
      models: ["model-1", "model-2"],
      max_budget: 150.75,
      tpm_limit: undefined,
      rpm_limit: undefined,
      budget_duration: "7d",
    };

    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expected);
  });

  it("sends null when the budget is cleared", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    const maxBudgetInput = await screen.findByLabelText("Max Budget (USD)");
    fireEvent.change(maxBudgetInput, { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining({ max_budget: null }));
  });

  it("sends zero as a budget value", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    fireEvent.change(await screen.findByLabelText("Max Budget (USD)"), { target: { value: "0" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining({ max_budget: 0 }));
  });

  it("sends the prefilled budget when it is left unchanged", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    expect(await screen.findByLabelText("Max Budget (USD)")).toHaveValue(10);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expect.objectContaining({ max_budget: 10 }));
  });

  it("blocks negative budgets before sending the request", async () => {
    const { user } = await renderEditor();
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    fireEvent.change(await screen.findByLabelText("Max Budget (USD)"), { target: { value: "-1" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    expect(await screen.findByText("Enter a nonnegative budget")).toBeInTheDocument();
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
  });

  it("should block the save when the tag name is cleared", async () => {
    const { user, nameInput } = await renderEditor();

    await user.clear(nameInput);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Please input a tag name")).toBeInTheDocument();
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
  });

  it("keeps a typed budget when the section is collapsed and reopened, as antd's store did", async () => {
    const { user } = await renderEditor();
    const toggle = () => screen.getByRole("button", { name: /Budget & Rate Limits/ });

    await user.click(toggle());
    const maxBudgetInput = await screen.findByLabelText("Max Budget (USD)");
    await user.clear(maxBudgetInput);
    fireEvent.change(maxBudgetInput, { target: { value: "150.75" } });

    await user.click(toggle());
    await user.click(toggle());

    expect(await screen.findByLabelText("Max Budget (USD)")).toHaveValue(150.75);

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    const expected = {
      name: "prod-tag",
      description: "original description",
      models: ["model-1", "model-2"],
      max_budget: 150.75,
      tpm_limit: undefined,
      rpm_limit: undefined,
      budget_duration: "7d",
    };

    expect(mockTagUpdateCall).toHaveBeenCalledWith("sk-test", expected);
  });

  it("leaves the tag untouched and returns to the detail view when Cancel is clicked", async () => {
    const { user } = await renderEditor();

    const descriptionInput = screen.getByLabelText("Description");
    await user.clear(descriptionInput);
    fireEvent.change(descriptionInput, { target: { value: "abandoned description" } });

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByText("Tag Details")).toBeInTheDocument();
    expect(mockTagUpdateCall).not.toHaveBeenCalled();
  });
});
