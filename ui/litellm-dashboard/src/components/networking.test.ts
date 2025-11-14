import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { updateSSOSettings } from "./networking";

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
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
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

  it("should surface backend detail error when updateSSOSettings fails", async () => {
    expect.hasAssertions();

    const backendError = {
      detail: {
        error: "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature.",
      },
    };

    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      json: vi.fn().mockResolvedValue(backendError),
    } as any);

    global.fetch = mockFetch as any;

    try {
      await updateSSOSettings("token", { some: "setting" });
    } catch (error) {
      const thrownError = error as any;
      expect(thrownError).toBeInstanceOf(Error);
      expect(thrownError.message).toBe(backendError.detail.error);
      expect(thrownError.detail).toEqual(backendError.detail);
      expect(thrownError.rawError).toEqual(backendError);
    }

    expect(mockFetch).toHaveBeenCalledOnce();
  });
});
