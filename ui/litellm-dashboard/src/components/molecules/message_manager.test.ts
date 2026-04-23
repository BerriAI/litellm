import { describe, expect, it, vi, beforeEach } from "vitest";

// Sonner is a module-level singleton; mock it before importing the manager.
const mockToast = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
  loading: vi.fn(),
  dismiss: vi.fn(),
}));

vi.mock("sonner", () => ({
  toast: mockToast,
}));

import MessageManager, { setMessageInstance } from "./message_manager";

describe("MessageManager", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("delegates to sonner", () => {
    it("delegates success", () => {
      MessageManager.success("done!");
      expect(mockToast.success).toHaveBeenCalledWith("done!", { duration: undefined });
    });

    it("delegates error with duration (converts seconds → ms)", () => {
      MessageManager.error("failed!", 5);
      expect(mockToast.error).toHaveBeenCalledWith("failed!", { duration: 5000 });
    });

    it("delegates warning", () => {
      MessageManager.warning("watch out");
      expect(mockToast.warning).toHaveBeenCalledWith("watch out", { duration: undefined });
    });

    it("delegates info with duration", () => {
      MessageManager.info("fyi", 2);
      expect(mockToast.info).toHaveBeenCalledWith("fyi", { duration: 2000 });
    });

    it("delegates loading and returns the toast id", () => {
      mockToast.loading.mockReturnValue("toast-id-42");
      const result = MessageManager.loading("loading...", 3);
      expect(mockToast.loading).toHaveBeenCalledWith("loading...", { duration: 3000 });
      expect(result).toBe("toast-id-42");
    });

    it("delegates destroy to toast.dismiss()", () => {
      MessageManager.destroy();
      expect(mockToast.dismiss).toHaveBeenCalled();
    });
  });

  describe("setMessageInstance is a no-op (back-compat shim)", () => {
    it("does not throw when called", () => {
      expect(() => setMessageInstance({} as unknown)).not.toThrow();
    });
  });
});
