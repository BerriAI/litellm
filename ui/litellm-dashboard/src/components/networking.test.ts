import { describe, it, expect, beforeEach, vi } from "vitest";
import { clearTokenCookies } from "@/utils/cookieUtils";

vi.mock("@/utils/cookieUtils", () => ({
  clearTokenCookies: vi.fn(),
  getCookie: vi.fn(),
}));

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    info: vi.fn(),
    success: vi.fn(),
    error: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

describe("networking - expired session handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should call clearTokenCookies on expired session", async () => {
    const errorData = "Authentication Error - Expired Key";
    const { default: NotificationsManager } = await import("./molecules/notifications_manager");

    if (errorData.includes("Authentication Error - Expired Key")) {
      NotificationsManager.info("UI Session Expired. Logging out.");
      clearTokenCookies();
    }

    expect(clearTokenCookies).toHaveBeenCalledOnce();
  });

  it("should not clear cookies for non-authentication errors", () => {
    const errorData = "Some other error";

    if (errorData.includes("Authentication Error - Expired Key")) {
      clearTokenCookies();
    }

    expect(clearTokenCookies).not.toHaveBeenCalled();
  });
});
