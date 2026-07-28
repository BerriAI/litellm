import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useValidateResetToken, useForgotPassword, useResetPassword } from "./usePasswordReset";
import * as networking from "@/components/networking";

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe("useForgotPassword", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls forgotPasswordCall with the given email", async () => {
    vi.spyOn(networking, "forgotPasswordCall").mockResolvedValue({ message: "ok" });
    const { result } = renderHook(() => useForgotPassword(), { wrapper });

    act(() => {
      result.current.mutate("alice@example.com");
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(networking.forgotPasswordCall).toHaveBeenCalledWith("alice@example.com");
  });

  it("exposes error state when the request fails", async () => {
    const error = new Error("Too many requests");
    vi.spyOn(networking, "forgotPasswordCall").mockRejectedValue(error);
    const { result } = renderHook(() => useForgotPassword(), { wrapper });

    act(() => {
      result.current.mutate("alice@example.com");
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toEqual(error);
  });
});

describe("useValidateResetToken", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not fetch when token is null", () => {
    const spy = vi.spyOn(networking, "validateResetTokenCall");
    renderHook(() => useValidateResetToken(null), { wrapper });
    expect(spy).not.toHaveBeenCalled();
  });

  it("fetches validation data when token is present", async () => {
    vi.spyOn(networking, "validateResetTokenCall").mockResolvedValue({ user_email: "alice@example.com" });
    const { result } = renderHook(() => useValidateResetToken("tok-123"), { wrapper });
    await waitFor(() => expect(result.current.data).toEqual({ user_email: "alice@example.com" }));
  });

  it("exposes error state when validation fails", async () => {
    const error = new Error("invalid token");
    vi.spyOn(networking, "validateResetTokenCall").mockRejectedValue(error);
    const { result } = renderHook(() => useValidateResetToken("bad-token"), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toEqual(error);
  });
});

describe("useResetPassword", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls resetPasswordCall with token and newPassword", async () => {
    vi.spyOn(networking, "resetPasswordCall").mockResolvedValue({ message: "ok" });
    const { result } = renderHook(() => useResetPassword(), { wrapper });

    act(() => {
      result.current.mutate({ token: "tok-123", newPassword: "new-secret" });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(networking.resetPasswordCall).toHaveBeenCalledWith("tok-123", "new-secret");
  });

  it("exposes error state when the request fails", async () => {
    const error = new Error("Password reset failed");
    vi.spyOn(networking, "resetPasswordCall").mockRejectedValue(error);
    const { result } = renderHook(() => useResetPassword(), { wrapper });

    act(() => {
      result.current.mutate({ token: "tok-123", newPassword: "new-secret" });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toEqual(error);
  });
});
