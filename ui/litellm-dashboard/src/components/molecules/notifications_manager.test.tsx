import { notification } from "antd";
import { beforeEach, describe, expect, it, vi } from "vitest";
import NotificationManager from "./notifications_manager";

vi.mock("@/components/molecules/notifications_manager", async () => {
  const actual = await vi.importActual<typeof import("@/components/molecules/notifications_manager")>(
    "@/components/molecules/notifications_manager",
  );

  return actual;
});

// Mock the antd notification module
vi.mock("antd", () => ({
  notification: {
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
    destroy: vi.fn(),
  },
}));

describe("NotificationManager", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Already Exists case", () => {
    it("should show error notification for 'already exists' message", () => {
      const error = {
        message: "Key with alias 'test10' already exists.",
        type: "bad_request_error",
        code: "400",
      };

      NotificationManager.fromBackend(error);

      expect(notification.error).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Already Exists",
          description: "Key with alias 'test10' already exists.",
          duration: 6,
          placement: "topRight",
        }),
      );
    });
  });
});
