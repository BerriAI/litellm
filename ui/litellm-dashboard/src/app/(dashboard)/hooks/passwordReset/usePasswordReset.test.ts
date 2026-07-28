import { describe, it, expect, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useValidateResetToken } from "./usePasswordReset";
import * as networking from "@/components/networking";

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe("useValidateResetToken", () => {
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
});
