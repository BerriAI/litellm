import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/http/client";
import { shouldRetry } from "./ReactQueryProvider";

const attempts = async (error: Error) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: shouldRetry, retryDelay: 0 } } });
  const queryFn = vi.fn().mockRejectedValue(error);
  await expect(client.fetchQuery({ queryKey: ["q"], queryFn })).rejects.toBe(error);
  return queryFn.mock.calls.length;
};

describe("root query retry policy", () => {
  it("does not retry client errors", async () => {
    expect(await attempts(new ApiError("missing", 404, null))).toBe(1);
    expect(await attempts(new ApiError("forbidden", 403, null))).toBe(1);
  });

  it("retries server and network errors", async () => {
    expect(await attempts(new ApiError("down", 503, null))).toBe(4);
    expect(await attempts(new TypeError("Failed to fetch"))).toBe(4);
  });
});
