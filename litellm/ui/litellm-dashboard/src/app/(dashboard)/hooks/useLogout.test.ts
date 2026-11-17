import { beforeEach, describe, expect, it, vi } from "vitest";

const sessionLogoutCall = vi.hoisted(() => vi.fn());
const clearTokenCookies = vi.hoisted(() => vi.fn());
const clearStoredReturnUrl = vi.hoisted(() => vi.fn());

vi.mock("@/components/networking", () => ({
  sessionLogoutCall,
}));
vi.mock("@/utils/cookieUtils", () => ({
  clearTokenCookies,
}));
vi.mock("@/utils/returnUrlUtils", () => ({
  clearStoredReturnUrl,
}));
vi.mock("@/app/(dashboard)/hooks/proxySettings/useProxySettings", () => ({
  default: vi.fn(() => ({ PROXY_LOGOUT_URL: "" })),
}));

import { revokeSessionAndClearClientState } from "./useLogout";

describe("revokeSessionAndClearClientState", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionLogoutCall.mockResolvedValue({ message: "Session revoked." });
    localStorage.setItem("litellm_selected_worker_id", "w1");
    localStorage.setItem("litellm_worker_url", "https://worker.example");
  });

  it("revokes the session server-side before clearing the token cookie", async () => {
    const order: string[] = [];
    sessionLogoutCall.mockImplementation(async () => {
      order.push("revoke");
      return { message: "Session revoked." };
    });
    clearTokenCookies.mockImplementation(() => {
      order.push("clearCookies");
    });

    await revokeSessionAndClearClientState("sk-token");

    expect(sessionLogoutCall).toHaveBeenCalledWith("sk-token");
    // The cookie holds the credential that authenticates the revoke call, so
    // clearing it first would orphan the server-side key.
    expect(order).toEqual(["revoke", "clearCookies"]);
  });

  it("clears all client state", async () => {
    await revokeSessionAndClearClientState("sk-token");

    expect(clearTokenCookies).toHaveBeenCalled();
    expect(clearStoredReturnUrl).toHaveBeenCalled();
    expect(localStorage.getItem("litellm_selected_worker_id")).toBeNull();
    expect(localStorage.getItem("litellm_worker_url")).toBeNull();
  });

  it("still clears client state when the revoke call rejects", async () => {
    sessionLogoutCall.mockRejectedValue(new Error("proxy unreachable"));

    await revokeSessionAndClearClientState("sk-token");

    expect(clearTokenCookies).toHaveBeenCalled();
    expect(localStorage.getItem("litellm_selected_worker_id")).toBeNull();
  });

  it("skips the server call without a token but still clears client state", async () => {
    await revokeSessionAndClearClientState(null);

    expect(sessionLogoutCall).not.toHaveBeenCalled();
    expect(clearTokenCookies).toHaveBeenCalled();
  });
});
