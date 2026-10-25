import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { toast } from "@/lib/toast";
import { convertPromptFileToJson, createPromptCall } from "@/components/networking";

import AddPromptForm from "./add_prompt_form";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { translate } from "@/i18n/language";
import { Toaster } from "@/components/ui/sonner";

vi.unmock("@/lib/toast");

vi.mock("@/components/networking", () => ({
  convertPromptFileToJson: vi.fn(),
  createPromptCall: vi.fn(),
}));

const mockConvert = vi.mocked(convertPromptFileToJson);
const mockCreate = vi.mocked(createPromptCall);

const PROMPT_ID_PLACEHOLDER = "Enter unique prompt ID (e.g., my_prompt_id)";

const CONVERTED_JSON = { model: "gpt-4o", messages: [{ role: "user", content: "hi {{name}}" }] };

const renderForm = (accessToken: string | null = "sk-test") => {
  const onClose = vi.fn();
  const onSuccess = vi.fn();
  render(
    <LanguageProvider defaultLanguage="en" userId="prompt-notices" storageScope="test">
      <LanguageSelector />
      <AddPromptForm visible onClose={onClose} accessToken={accessToken} onSuccess={onSuccess} />
      <Toaster expand />
    </LanguageProvider>,
  );
  return { onClose, onSuccess };
};

const typePromptId = (value: string) =>
  fireEvent.change(screen.getByPlaceholderText(PROMPT_ID_PLACEHOLDER), { target: { value } });

const attachPromptFile = async (file: File) => {
  const fileInput = screen.getByLabelText("Prompt file");
  await act(async () => {
    fireEvent.change(fileInput, { target: { files: [file] } });
  });
  await screen.findByText(`Selected: ${file.name}`);
};

const submit = async () => {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Create Prompt" }));
  });
};

describe("AddPromptForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockConvert.mockResolvedValue({ prompt_id: "converted_prompt_id", json_data: CONVERTED_JSON });
    mockCreate.mockResolvedValue({ status: "success" });
  });

  afterEach(() => act(() => toast.dismiss()));

  // 本地拒绝分支也使用真实通知，既不上传文件，也不因切换语言重复请求。
  it.each(["wrong-file", "missing-token"] as const)("%s 的本地通知随语言切换", async (reason) => {
    const { onClose, onSuccess } = renderForm(reason === "missing-token" ? null : "sk-test");
    typePromptId("Home");
    if (reason === "missing-token") {
      await attachPromptFile(new File(["Home"], "Home.prompt", { type: "text/plain" }));
      await submit();
    } else {
      fireEvent.change(screen.getByLabelText("Prompt file"), {
        target: { files: [new File(["Home"], "Home.txt", { type: "text/plain" })] },
      });
    }
    expect(
      await screen.findByText(reason === "missing-token" ? "Access token is required" : "Please upload a .prompt file"),
    ).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByText(reason === "missing-token" ? "需要访问令牌" : "请上传 .prompt 文件")).toBeInTheDocument();
    expect(screen.getByLabelText("提示词 ID")).toHaveValue("Home");
    expect(mockConvert).not.toHaveBeenCalled();
    expect(mockCreate).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
  });

  // 已存入表单的本地错误必须按显示时的语言翻译，而非要求再次提交。
  it.each([
    ["", "Please enter a prompt ID", "请输入提示词 ID"],
    [
      "Home invalid",
      "Prompt ID can only contain letters, numbers, underscores, and hyphens",
      "提示词 ID 只能包含字母、数字、下划线和连字符",
    ],
  ])("已有提示词校验随语言切换：%s", async (value, original, translated) => {
    renderForm();
    typePromptId(value);
    const file = new File(["Home"], "Home.prompt", { type: "text/plain" });
    await attachPromptFile(file);
    await submit();
    expect(await screen.findByText(original)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByText(translated)).toBeInTheDocument();
    expect(screen.queryByText(original)).not.toBeInTheDocument();
    expect(screen.getByLabelText("提示词 ID")).toHaveValue(value);
    expect(screen.getByText(/Home\.prompt/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(screen.getByText(original)).toBeInTheDocument();
    expect(mockConvert).not.toHaveBeenCalled();
    expect(mockCreate).not.toHaveBeenCalled();
  });

  // 转换、创建阶段分别延迟返回，确保旧回调不会固定通知语言。
  it.each(["success", "convert-failure", "create-failure"] as const)("%s 的异步通知跟随当前语言", async (result) => {
    const conversion = Promise.withResolvers<Awaited<ReturnType<typeof convertPromptFileToJson>>>();
    const creation = Promise.withResolvers<Awaited<ReturnType<typeof createPromptCall>>>();
    mockConvert.mockReturnValue(conversion.promise);
    mockCreate.mockReturnValue(creation.promise);
    const { onClose, onSuccess } = renderForm();
    typePromptId("Home");
    const file = new File(["Home"], "Home.prompt", { type: "text/plain" });
    await attachPromptFile(file);
    await submit();
    await waitFor(() => expect(mockConvert).toHaveBeenCalledTimes(1));
    if (result !== "convert-failure") {
      await act(async () => conversion.resolve({ prompt_id: "Home", json_data: { content: "Home" } }));
      await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    }
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    await act(async () => {
      if (result === "convert-failure") conversion.reject(new Error("test-only-internal"));
      else if (result === "create-failure") creation.reject(new Error("test-only-internal"));
      else creation.resolve({ status: "success" });
    });
    const messages = {
      success: ["提示词已创建！", "Prompt created successfully!"],
      "convert-failure": ["提示词文件转换为 JSON 失败", "Failed to convert prompt file to JSON"],
      "create-failure": ["创建提示词失败", "Failed to create prompt"],
    };
    expect(await screen.findByText(messages[result][0])).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(screen.getByText(messages[result][1])).toBeInTheDocument();
    expect(screen.queryByText("test-only-internal")).not.toBeInTheDocument();
    expect(mockConvert).toHaveBeenCalledExactlyOnceWith("sk-test", file);
    if (result === "convert-failure") expect(mockCreate).not.toHaveBeenCalled();
    else
      expect(mockCreate).toHaveBeenCalledExactlyOnceWith("sk-test", {
        prompt_id: "Home",
        litellm_params: { prompt_integration: "dotprompt", prompt_id: "Home", prompt_data: { content: "Home" } },
        prompt_info: { prompt_type: "db" },
      });
    expect(onClose).toHaveBeenCalledTimes(result === "success" ? 1 : 0);
    expect(onSuccess).toHaveBeenCalledTimes(result === "success" ? 1 : 0);
    if (result !== "success") {
      expect(screen.getByLabelText("Prompt ID")).toHaveValue("Home");
      expect(screen.getByText(/Home\.prompt/)).toBeInTheDocument();
    }
  });

  it("切换语言保留提示词 ID 和已选文件，并显示中文校验", async () => {
    localStorage.clear();
    render(
      <LanguageProvider defaultLanguage="en" userId="prompt-validation" storageScope="test">
        <LanguageSelector />
        <AddPromptForm visible onClose={vi.fn()} accessToken="sk-test" onSuccess={vi.fn()} />
      </LanguageProvider>,
    );
    typePromptId("Home invalid");
    const file = new File(["model: gpt-4o"], "Home.prompt", { type: "text/plain" });
    await attachPromptFile(file);

    // 文件名和提示词内容属于用户数据，不跟随界面语言翻译或重置。
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByDisplayValue("Home invalid")).toBeInTheDocument();
    expect(screen.getByText(/Home\.prompt/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: translate("zh-CN", "Create Prompt") }));
    expect(await screen.findByText("提示词 ID 只能包含字母、数字、下划线和连字符")).toBeInTheDocument();
    expect(mockCreate).not.toHaveBeenCalled();
    expect(mockConvert).not.toHaveBeenCalled();
  });

  it("sends the converted upload as the exact create-prompt payload, then closes and refreshes", async () => {
    const { onClose, onSuccess } = renderForm();
    const file = new File(["model: gpt-4o"], "greeting.prompt", { type: "text/plain" });

    typePromptId("my_prompt_id");
    await attachPromptFile(file);
    await submit();

    await waitFor(() => {
      expect(mockCreate).toHaveBeenCalledWith("sk-test", {
        prompt_id: "my_prompt_id",
        litellm_params: {
          prompt_integration: "dotprompt",
          prompt_id: "converted_prompt_id",
          prompt_data: CONVERTED_JSON,
        },
        prompt_info: {
          prompt_type: "db",
        },
      });
    });
    expect(mockConvert).toHaveBeenCalledWith("sk-test", file);
    expect(await screen.findByText("Prompt created successfully!")).toBeInTheDocument();
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onSuccess).toHaveBeenCalledTimes(1);
  });

  it("refuses to submit without an uploaded file", async () => {
    renderForm();

    typePromptId("my_prompt_id");
    await submit();

    expect(await screen.findByText("Please upload a .prompt file")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    expect(screen.getByText("请上传 .prompt 文件")).toBeInTheDocument();
    expect(mockConvert).not.toHaveBeenCalled();
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("blocks submission and reports a missing prompt ID", async () => {
    renderForm();

    await submit();

    expect(await screen.findByText("Please enter a prompt ID")).toBeInTheDocument();
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("blocks submission and reports a prompt ID with unsupported characters", async () => {
    renderForm();
    const file = new File(["model: gpt-4o"], "greeting.prompt", { type: "text/plain" });

    typePromptId("my prompt!");
    await attachPromptFile(file);
    await submit();

    expect(
      await screen.findByText("Prompt ID can only contain letters, numbers, underscores, and hyphens"),
    ).toBeInTheDocument();
    expect(mockCreate).not.toHaveBeenCalled();
  });
});
