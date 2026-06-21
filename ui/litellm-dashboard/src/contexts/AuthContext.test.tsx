/* @vitest-environment jsdom */
import React from "react";
import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthProvider, useAuth } from "./AuthContext";

// getUiConfig never resolves here on purpose: it lets the test prove that auth state is
// available from the first render, independent of the config fetch. The bug was that userID
// only became available inside an effect gated behind this await, so a consumer could render
// with a valid token but a null userID ("User ID is not set").
vi.mock("@/components/networking", () => ({
  getUiConfig: vi.fn(() => new Promise<never>(() => {})),
  setGlobalLitellmHeaderName: vi.fn(),
}));

const base64Url = (obj: Record<string, unknown>) =>
  btoa(JSON.stringify(obj)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

const makeToken = (payload: Record<string, unknown>) =>
  `${base64Url({ alg: "HS256", typ: "JWT" })}.${base64Url(payload)}.sig`;

const setTokenCookie = (token: string) => {
  document.cookie = `token=${token}; path=/`;
};

const inOneHour = () => Math.floor(Date.now() / 1000) + 3600;

const wrapper = ({ children }: { children: React.ReactNode }) => <AuthProvider>{children}</AuthProvider>;

describe("AuthProvider", () => {
  afterEach(() => {
    document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 UTC";
    vi.clearAllMocks();
  });

  it("exposes auth state from a valid cookie on first render, before getUiConfig resolves", () => {
    setTokenCookie(
      makeToken({
        user_id: "default_user_id",
        user_role: "proxy_admin",
        key: "sk-test-key",
        login_method: "username_password",
        exp: inOneHour(),
      }),
    );

    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.userID).toBe("default_user_id");
    expect(result.current.accessToken).toBe("sk-test-key");
    expect(result.current.userRole).toBe("Admin");
  });

  it("leaves userID null when no token cookie is present", () => {
    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.userID).toBeNull();
    expect(result.current.accessToken).toBeNull();
  });

  it("leaves userID null when the token cookie is expired", () => {
    setTokenCookie(
      makeToken({
        user_id: "default_user_id",
        user_role: "proxy_admin",
        key: "sk-test-key",
        exp: Math.floor(Date.now() / 1000) - 60,
      }),
    );

    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.userID).toBeNull();
  });
});
