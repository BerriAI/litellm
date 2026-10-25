import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { translate, type Language } from "@/i18n/language";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";

import PromptCompressionTab from "./PromptCompressionTab";

const createGuardrailCall = vi.fn();
const getGuardrailsList = vi.fn();

vi.unmock("@/lib/toast");

vi.mock("@/components/networking", () => ({
  createGuardrailCall: (...args: unknown[]) => createGuardrailCall(...args),
  getGuardrailsList: (...args: unknown[]) => getGuardrailsList(...args),
}));

const submittedPayload = (): Record<string, unknown> => {
  expect(createGuardrailCall).toHaveBeenCalledTimes(1);
  return createGuardrailCall.mock.calls[0][1] as Record<string, unknown>;
};

describe("PromptCompressionTab submit payload", () => {
  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
    createGuardrailCall.mockClear().mockResolvedValue({});
    getGuardrailsList.mockClear().mockResolvedValue({ guardrails: [] });
  });

  afterEach(() => act(() => toast.dismiss()));

  it("压缩列表加载失败后切换语言更新现有通知且不重试请求", async () => {
    getGuardrailsList.mockRejectedValue(new Error("Home"));
    render(
      <LanguageProvider defaultLanguage="en" userId="compression-load-notice" storageScope="compression-load-notice">
        <LanguageSelector />
        <PromptCompressionTab accessToken="test-token" />
        <Toaster expand />
      </LanguageProvider>,
    );
    expect(await screen.findByText("Failed to load compression guardrails")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(await screen.findByText(translate("zh-CN", "Failed to load compression guardrails"))).toBeVisible();
    expect(screen.queryByText("Failed to load compression guardrails")).not.toBeInTheDocument();
    expect(getGuardrailsList).toHaveBeenCalledTimes(1);
  });

  it.each(["success", "failure"])("创建请求的 %s 通知在等待与展示后切换语言时同步更新", async (result) => {
    const user = userEvent.setup();
    const request = Promise.withResolvers<unknown>();
    createGuardrailCall.mockReturnValue(request.promise);
    render(
      <LanguageProvider defaultLanguage="en" userId="compression-notice" storageScope="compression-notice">
        <LanguageSelector />
        <PromptCompressionTab accessToken="test-token" />
        <Toaster expand />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("Headroom API base"), { target: { value: "https://example.com/Settings" } });
    await user.click(screen.getByRole("button", { name: "Add guardrail" }));
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    await act(async () => {
      if (result === "success") request.resolve({});
      else request.reject(new Error("Home"));
    });
    const message = result === "success" ? "Compression guardrail created" : "Failed to create compression guardrail";
    expect(await screen.findByText(translate("zh-CN", message))).toBeVisible();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(await screen.findByText(message)).toBeVisible();
    expect(screen.queryByText(translate("zh-CN", message))).not.toBeInTheDocument();
    expect(createGuardrailCall).toHaveBeenCalledTimes(1);
    expect(getGuardrailsList).toHaveBeenCalledTimes(result === "success" ? 2 : 1);
    if (result === "failure") expect(screen.getByLabelText("Name")).toHaveValue("Home");
  });

  it.each<Language>(["en", "zh-CN"])("压缩表单已显示的错误随 %s 切换且不重新加载", async (initialLanguage) => {
    const user = userEvent.setup();
    const nextLanguage = initialLanguage === "en" ? "zh-CN" : "en";
    render(
      <LanguageProvider defaultLanguage={initialLanguage} userId="compression-switch" storageScope="compression-switch">
        <LanguageSelector />
        <PromptCompressionTab accessToken="test-token" />
      </LanguageProvider>,
    );
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Add guardrail") }));
    expect(await screen.findByText(translate(initialLanguage, "Name is required"))).toBeVisible();
    expect(screen.getByText(translate(initialLanguage, "API base is required"))).toBeVisible();
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Interface language")), {
      target: { value: nextLanguage },
    });
    expect(screen.getByText(translate(nextLanguage, "Name is required"))).toBeVisible();
    expect(screen.getByText(translate(nextLanguage, "API base is required"))).toBeVisible();
    expect(createGuardrailCall).not.toHaveBeenCalled();
    expect(getGuardrailsList).toHaveBeenCalledTimes(1);
    fireEvent.change(screen.getByLabelText(translate(nextLanguage, "Name")), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText(translate(nextLanguage, "Headroom API base")), {
      target: { value: "https://example.com/Settings" },
    });
    await user.click(screen.getByRole("button", { name: translate(nextLanguage, "Add guardrail") }));
    expect(createGuardrailCall).toHaveBeenCalledWith("test-token", {
      guardrail_name: "Home",
      litellm_params: {
        guardrail: "headroom",
        mode: "pre_call",
        api_base: "https://example.com/Settings",
        default_on: true,
      },
    });
  });

  it("sends the trimmed name and api base with default_on true", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "  headroom-compression  " } });
    fireEvent.change(screen.getByLabelText("Headroom API base"), {
      target: { value: "  https://headroom.example.com  " },
    });
    await user.click(screen.getByRole("button", { name: "Add guardrail" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        guardrail_name: "headroom-compression",
        litellm_params: {
          guardrail: "headroom",
          mode: "pre_call",
          api_base: "https://headroom.example.com",
          default_on: true,
        },
      }),
    );
    expect(createGuardrailCall.mock.calls[0][0]).toBe("test-token");
  });

  it("sends default_on false once the apply-to-all switch is turned off", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "headroom-optin" } });
    fireEvent.change(screen.getByLabelText("Headroom API base"), { target: { value: "https://headroom.example.com" } });
    await user.click(screen.getByLabelText("Apply to all requests"));
    await user.click(screen.getByRole("button", { name: "Add guardrail" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        guardrail_name: "headroom-optin",
        litellm_params: {
          guardrail: "headroom",
          mode: "pre_call",
          api_base: "https://headroom.example.com",
          default_on: false,
        },
      }),
    );
  });

  it("blocks submission and shows both required messages when the form is empty", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    await user.click(screen.getByRole("button", { name: "Add guardrail" }));

    expect(await screen.findByText("Name is required")).toBeInTheDocument();
    expect(screen.getByText("API base is required")).toBeInTheDocument();
    expect(createGuardrailCall).not.toHaveBeenCalled();
  });

  it("submits when Enter is pressed inside a text field", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "headroom-compression" } });
    await user.type(screen.getByLabelText("Headroom API base"), "https://headroom.example.com{Enter}");

    await vi.waitFor(() => expect(createGuardrailCall).toHaveBeenCalledTimes(1));
  });

  it("clears the name and restores the default switch state after a successful create", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "headroom-compression" } });
    fireEvent.change(screen.getByLabelText("Headroom API base"), { target: { value: "https://headroom.example.com" } });
    await user.click(screen.getByLabelText("Apply to all requests"));
    await user.click(screen.getByRole("button", { name: "Add guardrail" }));

    await vi.waitFor(() => expect(screen.getByLabelText("Name")).toHaveValue(""));
    expect(screen.getByLabelText("Headroom API base")).toHaveValue("");
    expect(screen.getByLabelText("Apply to all requests")).toBeChecked();
    expect(getGuardrailsList).toHaveBeenCalledTimes(2);
  });

  it("keeps the always-on and opt-in badges for the guardrails it lists", async () => {
    getGuardrailsList.mockResolvedValue({
      guardrails: [
        {
          guardrail_id: "g-1",
          guardrail_name: "always-on-one",
          litellm_params: { guardrail: "headroom", api_base: "https://a.example.com", default_on: true },
        },
        {
          guardrail_id: "g-2",
          guardrail_name: "opt-in-one",
          litellm_params: { guardrail: "headroom", api_base: "https://b.example.com", default_on: false },
        },
      ],
    });
    render(<PromptCompressionTab accessToken="test-token" />);

    expect(await screen.findByText("Always on")).toBeInTheDocument();
    expect(screen.getByText("Opt-in")).toBeInTheDocument();
    expect(screen.getByText("https://a.example.com")).toBeInTheDocument();
  });
});
