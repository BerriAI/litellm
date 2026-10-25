import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import {
  deleteConfigFieldSetting,
  handleError,
  keyInfoV1Call,
  modelCreateCall,
  modelInfoCall,
  teamUpdateCall,
  updateConfigFieldSetting,
  updatePassThroughEndpoint,
} from "./networking";

vi.unmock("@/lib/toast");

function renderNotifications() {
  render(
    <LanguageProvider defaultLanguage="zh-CN" userId="networking-language" storageScope="test">
      <Toaster expand />
      <LanguageSelector />
    </LanguageProvider>,
  );
}

// 网络请求与通知展示走真实实现，仅替换 fetch 边界，不翻用户输入或后端详情。
describe("网络层固定通知语言", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({ saved: true }));
    renderNotifications();
  });

  afterEach(() => {
    act(() => toast.dismiss());
    vi.restoreAllMocks();
  });

  it("翻译配置更新通知而不改变配置项和值", async () => {
    await act(async () => {
      await updateConfigFieldSetting("test-token", "Home", "Settings");
    });
    expect(await screen.findByText("设置值已更新！")).toBeVisible();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/config/field/update"),
      expect.objectContaining({
        body: JSON.stringify({ field_name: "Home", field_value: "Settings", config_type: "general_settings" }),
      }),
    );
  });

  it("翻译配置重置通知", async () => {
    await act(async () => {
      await deleteConfigFieldSetting("test-token", "Home");
    });
    expect(await screen.findByText("已重置代理上的配置项")).toBeVisible();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/config/field/delete"),
      expect.objectContaining({ body: JSON.stringify({ field_name: "Home", config_type: "general_settings" }) }),
    );
  });

  it("翻译透传端点更新通知而不改变路径和提交内容", async () => {
    await act(async () => {
      await updatePassThroughEndpoint("test-token", "Home", { description: "Settings" });
    });
    expect(await screen.findByText("透传端点已更新")).toBeVisible();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/config/pass_through_endpoint/Home"),
      expect.objectContaining({ body: JSON.stringify({ description: "Settings" }) }),
    );
  });

  it("翻译模型创建模板且切回英文保留模型名", async () => {
    const model = { model_name: "Home", litellm_params: { model: "Settings" }, model_info: null };
    await act(async () => {
      await modelCreateCall("test-token", model);
    });
    expect(await screen.findByText("模型 Home 创建成功")).toBeVisible();
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(await screen.findByText("Model Home created successfully")).toBeVisible();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/model/new"),
      expect.objectContaining({ body: JSON.stringify(model) }),
    );
  });

  it("翻译密钥查询失败前缀，保留后端详情与权限错误分类", async () => {
    vi.mocked(fetch).mockResolvedValue(
      Response.json({ error: { message: "Home", type: "team_model_access_denied", code: "403" } }, { status: 403 }),
    );
    await act(async () => {
      // 原实现报错响应会再次读取body，本轮仅验证通知，不改变既有异常行为。
      await expect(keyInfoV1Call("test-token", "Settings")).rejects.toThrow();
    });
    expect(await screen.findByText("获取密钥信息失败：Home")).toBeVisible();
    expect(screen.getByText("无访问权限")).toBeVisible();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
  });

  it("翻译团队更新失败前缀，但抛出的原始错误和业务值不变", async () => {
    const raw = JSON.stringify({ error: { message: "Home", type: "bad_request_error", code: "400" } });
    vi.mocked(fetch).mockResolvedValue(new Response(raw, { status: 400 }));
    await act(async () => {
      await expect(teamUpdateCall("test-token", { team_id: "Settings" })).rejects.toThrow(raw);
    });
    expect(await screen.findByText("更新团队设置失败：Home")).toBeVisible();
    // 原调用先展开服务端错误，保留其不携带状态码时的通用错误分类。
    expect(screen.getByText("错误")).toBeVisible();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
  });

  it("仅本地无模型提示分支翻译为中文", async () => {
    vi.mocked(fetch).mockResolvedValue(new Response("No model list passed", { status: 400 }));
    await act(async () => {
      await expect(modelInfoCall("test-token", "user-1", "Admin")).rejects.toThrow("Network response was not ok");
    });
    expect(await screen.findByText("尚无模型，请点击“添加模型”开始。")).toBeVisible();
  });

  it("翻译会话过期通知，保留原有退出处理", async () => {
    vi.spyOn(Date, "now").mockReturnValue(7_000_000_000_000);
    await act(async () => {
      await handleError("Authentication Error - Expired Key");
    });
    expect(await screen.findByText("界面会话已过期，正在退出登录。")).toBeVisible();
  });
});
