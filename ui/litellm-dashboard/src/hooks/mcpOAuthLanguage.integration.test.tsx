import { act, renderHook, waitFor } from "@testing-library/react";
import type { PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { translate } from "@/i18n/language";
import { toast } from "@/lib/toast";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";
import { useMcpOAuthFlow } from "./useMcpOAuthFlow";
import { useToolsOAuthFlow } from "./useToolsOAuthFlow";
import { useUserMcpOAuthFlow } from "./useUserMcpOAuthFlow";

vi.mock("@/components/networking", () => ({
  exchangeMcpOAuthToken: vi.fn(),
  cacheTemporaryMcpServer: vi.fn(),
  registerMcpOAuthClient: vi.fn(),
  storeMCPOAuthUserCredential: vi.fn(),
  buildMcpOAuthAuthorizeUrl: vi.fn(),
  getProxyBaseUrl: () => "",
  serverRootPath: "",
}));

const onSuccess = vi.fn();
const credentials = () => ({});
const payload = () => ({ url: "https://server.example/mcp", transport: "http" });
const adminOptions = {
  accessToken: "test-caller",
  getCredentials: credentials,
  getTemporaryPayload: payload,
  onTokenReceived: onSuccess,
  flowSource: "create",
};

function useAdminFlow() {
  return useMcpOAuthFlow(adminOptions);
}

function useUserFlow() {
  return useUserMcpOAuthFlow({ accessToken: "test-caller", serverId: "server-1", onSuccess });
}

function useToolsFlow() {
  const options = { accessToken: "test-caller", serverId: "server-1", userId: "user-1", onSuccess };
  return useToolsOAuthFlow(options);
}

function EnglishProvider({ children }: PropsWithChildren) {
  return (
    <LanguageProvider defaultLanguage="en" userId="oauth-language" storageScope="oauth-test">
      {children}
    </LanguageProvider>
  );
}

function renderFlow(useFlow: typeof useAdminFlow | typeof useUserFlow | typeof useToolsFlow) {
  return renderHook(() => ({ flow: useFlow(), language: useLanguage() }), { wrapper: EnglishProvider });
}

const flows = [
  { name: "admin", prefix: "litellm-mcp-oauth", useFlow: useAdminFlow, success: "OAuth token retrieved successfully" },
  { name: "user", prefix: "litellm-user-mcp-oauth", useFlow: useUserFlow, success: "Connected successfully" },
  { name: "tools", prefix: "litellm-tools-mcp-oauth", useFlow: useToolsFlow, success: "Connected successfully" },
] as const;

function seedFlow(prefix: string, result: Record<string, string> = { state: "state-1", code: "code-1" }) {
  const flowState = {
    state: "state-1",
    codeVerifier: "verifier-1",
    serverId: "server-1",
    redirectUri: "https://app.example/callback",
    clientId: "client-1",
    flowSource: "create",
  };
  setSecureItem(`${prefix}-flow-state`, JSON.stringify(flowState));
  setSecureItem(`${prefix}-result`, JSON.stringify(result));
}

describe("MCP OAuth 通知语言", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    localStorage.clear();
  });

  it.each(flows)("$name 交换过程中切语言不重发请求或清草稿，成功使用最新语言", async ({ prefix, useFlow, success }) => {
    let resolveExchange: (value: { access_token: string }) => void = () => undefined;
    const pending = new Promise<{ access_token: string }>((resolve) => {
      resolveExchange = resolve;
    });
    vi.mocked(networking.exchangeMcpOAuthToken).mockReturnValueOnce(pending);
    seedFlow(prefix);
    setSecureItem("mcp-form-draft", "Home");
    const { result } = renderFlow(useFlow);
    await waitFor(() => expect(result.current.flow.status).toBe("exchanging"));
    const startFlow = result.current.flow.startOAuthFlow;
    const flowState = getSecureItem(`${prefix}-flow-state`);

    act(() => result.current.language.setPreference("zh-CN"));

    expect(result.current.flow.startOAuthFlow).toBe(startFlow);
    expect(networking.exchangeMcpOAuthToken).toHaveBeenCalledTimes(1);
    expect(getSecureItem(`${prefix}-flow-state`)).toBe(flowState);
    expect(getSecureItem("mcp-form-draft")).toBe("Home");
    expect(localStorage.getItem(`${prefix}-flow-state`)).toBeNull();

    await act(async () => {
      resolveExchange({ access_token: "test-provider-token" });
      await pending;
    });

    await waitFor(() => expect(result.current.flow.status).toBe("success"));
    expect(toast.success).toHaveBeenCalledExactlyOnceWith(translate("zh-CN", success));
    expect(translate("zh-CN", success)).not.toBe(success);
    const expectedExchange = {
      serverId: "server-1",
      code: "code-1",
      codeVerifier: "verifier-1",
      accessToken: "test-caller",
    };
    expect(networking.exchangeMcpOAuthToken).toHaveBeenCalledExactlyOnceWith(expect.objectContaining(expectedExchange));
  });

  it("管理员本地校验错误可即时切换语言，不重复通知或授权", async () => {
    const { result } = renderHook(
      () => ({ flow: useMcpOAuthFlow({ ...adminOptions, accessToken: null }), language: useLanguage() }),
      { wrapper: EnglishProvider },
    );
    await act(async () => result.current.flow.startOAuthFlow());
    expect(result.current.flow.error).toBe("Missing admin token");

    act(() => result.current.language.setPreference("zh-CN"));

    expect(result.current.flow.error).toBe(translate("zh-CN", "Missing admin token"));
    expect(result.current.flow.error).not.toBe("Missing admin token");
    expect(toast.error).toHaveBeenCalledExactlyOnceWith("Access token missing. Please re-authenticate and try again.");
    expect(networking.cacheTemporaryMcpServer).not.toHaveBeenCalled();
    expect(networking.exchangeMcpOAuthToken).not.toHaveBeenCalled();
  });

  it("管理员回调的本地错误保持来源标记，语言切换不重跑回调", async () => {
    seedFlow("litellm-mcp-oauth", { state: "wrong-state", code: "code-1" });
    const { result } = renderFlow(useAdminFlow);
    await waitFor(() => expect(result.current.flow.status).toBe("error"));

    act(() => result.current.language.setPreference("zh-CN"));

    expect(result.current.flow.error).toBe(translate("zh-CN", "OAuth state mismatch. Please retry."));
    expect(result.current.flow.error).not.toBe("OAuth state mismatch. Please retry.");
    expect(toast.error).toHaveBeenCalledTimes(1);
    expect(networking.exchangeMcpOAuthToken).not.toHaveBeenCalled();
  });

  it("缺少服务地址时使用当前语言通知，切回英文只更新错误展示", async () => {
    const options = { ...adminOptions, getTemporaryPayload: () => null };
    const { result } = renderHook(() => ({ flow: useMcpOAuthFlow(options), language: useLanguage() }), {
      wrapper: EnglishProvider,
    });
    act(() => result.current.language.setPreference("zh-CN"));

    await act(async () => result.current.flow.startOAuthFlow());

    const message = "Please complete server URL and transport before starting OAuth.";
    expect(result.current.flow.error).toBe(translate("zh-CN", message));
    expect(result.current.flow.error).not.toBe(message);
    expect(toast.error).toHaveBeenCalledExactlyOnceWith(translate("zh-CN", message));

    act(() => result.current.language.setPreference("en"));

    expect(result.current.flow.error).toBe(message);
    expect(toast.error).toHaveBeenCalledTimes(1);
    expect(networking.cacheTemporaryMcpServer).not.toHaveBeenCalled();
  });

  it("损坏回调的固定恢复错误可翻译，不改变清理行为", async () => {
    seedFlow("litellm-mcp-oauth");
    setSecureItem("litellm-mcp-oauth-result", "{");
    const { result } = renderFlow(useAdminFlow);
    await waitFor(() => expect(result.current.flow.status).toBe("error"));

    act(() => result.current.language.setPreference("zh-CN"));

    const message = "Failed to resume OAuth flow. Please retry.";
    expect(result.current.flow.error).toBe(translate("zh-CN", message));
    expect(result.current.flow.error).not.toBe(message);
    expect(toast.error).toHaveBeenCalledExactlyOnceWith(message);
    expect(getSecureItem("litellm-mcp-oauth-flow-state")).toBeNull();
    expect(getSecureItem("litellm-mcp-oauth-result")).toBeNull();
    expect(networking.exchangeMcpOAuthToken).not.toHaveBeenCalled();
  });

  it("与词典命中相同的后端错误仍保留原文", async () => {
    const backendMessage = "OAuth state mismatch. Please retry.";
    seedFlow("litellm-mcp-oauth", { state: "state-1", error: "denied", error_description: backendMessage });
    const { result } = renderFlow(useAdminFlow);
    await waitFor(() => expect(result.current.flow.status).toBe("error"));

    act(() => result.current.language.setPreference("zh-CN"));

    expect(result.current.flow.error).toBe(backendMessage);
    expect(toast.error).toHaveBeenCalledExactlyOnceWith(backendMessage);
    expect(networking.exchangeMcpOAuthToken).not.toHaveBeenCalled();
  });
});
