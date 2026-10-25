import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// 本地缺省提示可以翻译，但后端返回的相同文案仍按原字符串传给通知层。
describe("模型列表后端通知原文", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it.each(["Home", "No Models Exist. Click Add Model to get started."])("保留后端文本 %s", async (message) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(message, { status: 400 }));
    const { modelInfoCall } = await import("./networking");
    const { toast } = await import("@/lib/toast");
    await expect(modelInfoCall("test-token", "user-1", "Admin")).rejects.toThrow("Network response was not ok");
    expect(toast.info).toHaveBeenCalledWith(`${message}error shown=false`);
  });
});
