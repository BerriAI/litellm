import { describe, expect, it, vi, beforeEach } from "vitest";

// Use vi.hoisted so the mock object is available when vi.mock is hoisted
const mockStaticMessage = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
  loading: vi.fn(),
  destroy: vi.fn(),
}));

vi.mock("antd", () => ({
  message: mockStaticMessage,
}));

import MessageManager, { setMessageInstance } from "./message_manager";

describe("MessageManager", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("when no instance is set (falls back to static message)", () => {
    it("delegates success to static message", () => {
      MessageManager.success("done!");
      expect(mockStaticMessage.success).toHaveBeenCalledWith("done!", undefined);
    });

    it("delegates error to static message", () => {
      MessageManager.error("failed!", 5);
      expect(mockStaticMessage.error).toHaveBeenCalledWith("failed!", 5);
    });

    it("delegates warning to static message", () => {
      MessageManager.warning("watch out");
      expect(mockStaticMessage.warning).toHaveBeenCalledWith("watch out", undefined);
    });

    it("delegates info to static message", () => {
      MessageManager.info("fyi");
      expect(mockStaticMessage.info).toHaveBeenCalledWith("fyi", undefined);
    });

    it("delegates loading to static message", () => {
      MessageManager.loading("loading...", 3);
      expect(mockStaticMessage.loading).toHaveBeenCalledWith("loading...", 3);
    });

    it("delegates destroy to static message", () => {
      MessageManager.destroy();
      expect(mockStaticMessage.destroy).toHaveBeenCalled();
    });
  });

  describe("when a custom instance is set", () => {
    const mockInstance = {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
      loading: vi.fn(),
      destroy: vi.fn(),
      open: vi.fn(),
    };

    beforeEach(() => {
      vi.clearAllMocks();
      setMessageInstance(mockInstance as any);
    });

    it("delegates success to custom instance", () => {
      MessageManager.success("done!");
      expect(mockInstance.success).toHaveBeenCalledWith("done!", undefined);
      expect(mockStaticMessage.success).not.toHaveBeenCalled();
    });

    it("delegates error with duration to custom instance", () => {
      MessageManager.error("failed!", 5);
      expect(mockInstance.error).toHaveBeenCalledWith("failed!", 5);
      expect(mockStaticMessage.error).not.toHaveBeenCalled();
    });

    it("delegates warning to custom instance", () => {
      MessageManager.warning("watch out");
      expect(mockInstance.warning).toHaveBeenCalledWith("watch out", undefined);
    });

    it("delegates info to custom instance", () => {
      MessageManager.info("fyi", 2);
      expect(mockInstance.info).toHaveBeenCalledWith("fyi", 2);
    });

    it("delegates loading to custom instance and returns result", () => {
      const mockReturn = { then: vi.fn() };
      mockInstance.loading.mockReturnValue(mockReturn);
      const result = MessageManager.loading("loading...", 3);
      expect(mockInstance.loading).toHaveBeenCalledWith("loading...", 3);
      expect(result).toBe(mockReturn);
    });

    it("delegates destroy to custom instance", () => {
      MessageManager.destroy();
      expect(mockInstance.destroy).toHaveBeenCalled();
    });
  });
});
