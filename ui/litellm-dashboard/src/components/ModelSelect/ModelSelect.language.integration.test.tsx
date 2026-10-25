import { useState } from "react";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { ModelSelect } from "./ModelSelect";

// 只替换 HTTP 边界，保留模型查询、真实选择器和语言上下文。
describe("模型选择器语言边界", () => {
  beforeEach(() => {
    localStorage.clear();
    testQueryClient.clear();
  });

  afterEach(() => vi.restoreAllMocks());

  it("不可用分组使用模型语境的翻译并保留已有选择", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const requests: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = new URL(input instanceof Request ? input.url : String(input), "http://localhost").pathname;
      requests.push(path);
      if (path.endsWith("/models")) return Response.json({ data: [] });
      if (path === "/v2/user/info") return Response.json({ models: [] });
      throw new Error(`未配置测试响应 ${path}`);
    });
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="unavailable-model" storageScope="test">
        <LanguageSelector />
        <ModelSelect context="organization" value={["Home"]} onChange={onChange} />
      </LanguageProvider>,
    );
    await user.click(await screen.findByRole("combobox", { name: "选择模型" }));
    expect(await screen.findByText("不可用模型")).toBeVisible();
    expect(screen.getByRole("option", { name: "Home" })).toBeVisible();
    expect(screen.queryByText("暂无数据")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(await screen.findByText("Unavailable Models")).toBeVisible();
    expect(onChange).not.toHaveBeenCalled();
    expect(requests).toHaveLength(2);
  });

  it.each(["all-proxy-models", "no-default-models", "Home/*"])(
    "%s 选项与分组翻译，业务名称和选项值保持原文",
    async (selection) => {
      const user = userEvent.setup();
      const onChange = vi.fn();
      const requests: string[] = [];
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const path = new URL(input instanceof Request ? input.url : String(input), "http://localhost").pathname;
        requests.push(path);
        if (path.endsWith("/models")) {
          return Response.json({
            data: ["Home", "Settings", "All Proxy Models", "Home/*"].map((id) => ({
              id,
              object: "model",
              created: 0,
              owned_by: "test-provider",
            })),
          });
        }
        if (path === "/v2/user/info") return Response.json({ models: [] });
        throw new Error(`未配置测试响应 ${path}`);
      });
      const Harness = () => {
        const [value, setValue] = useState<string[]>([]);
        return (
          <ModelSelect
            context="organization"
            options={{ includeSpecialOptions: true, showAllProxyModelsOverride: true }}
            value={value}
            onChange={(next) => {
              setValue(next);
              onChange(next);
            }}
          />
        );
      };
      renderWithProviders(
        <LanguageProvider defaultLanguage="en" userId="model-options" storageScope="test">
          <LanguageSelector />
          <Harness />
        </LanguageProvider>,
      );
      await screen.findByRole("combobox", { name: "Select Models" });
      await waitFor(() => expect(requests).toHaveLength(2));
      fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
      await user.click(screen.getByRole("combobox", { name: "选择模型" }));
      expect(await screen.findByText("特殊选项")).toBeVisible();
      expect(screen.getByText("通配符选项")).toBeVisible();
      expect(screen.getByRole("option", { name: "所有 Home 模型" })).toBeVisible();
      expect(screen.getByRole("option", { name: "所有代理模型", exact: true })).toBeVisible();
      expect(screen.getByRole("option", { name: "不设置默认模型" })).toBeVisible();
      expect(screen.getByRole("option", { name: "Home", exact: true })).toBeVisible();
      expect(screen.getByRole("option", { name: "Settings", exact: true })).toBeVisible();
      // 普通模型与固定选项恰好同名，也不能把普通模型当作翻译键。
      expect(screen.getByRole("option", { name: "All Proxy Models", exact: true })).toBeVisible();
      const labels: Record<string, string> = {
        "all-proxy-models": "所有代理模型",
        "no-default-models": "不设置默认模型",
        "Home/*": "所有 Home 模型",
      };
      await user.click(screen.getByRole("option", { name: labels[selection], exact: true }));
      expect(onChange).toHaveBeenCalledExactlyOnceWith([selection]);
      await user.keyboard("{Escape}");
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      await user.click(screen.getByRole("combobox", { name: "Select Models" }));
      expect(await screen.findByText("Special Options")).toBeVisible();
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(requests).toHaveLength(2);
    },
  );
});
