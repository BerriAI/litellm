import { act, renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { PropsWithChildren } from "react";
import { describe, expect, it, vi } from "vitest";

import { NEWEST } from "./list/runOrder";
import { useRunOrderRouting } from "./routing";

const renderRouting = (searchParams: string) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  const wrapper = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter searchParams={searchParams} onUrlUpdate={onUrlUpdate} hasMemory>
      {children}
    </NuqsTestingAdapter>
  );
  const hook = renderHook(() => useRunOrderRouting(), { wrapper });
  const lastUrl = () => new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString ?? "");
  return { ...hook, lastUrl, onUrlUpdate };
};

describe("useRunOrderRouting", () => {
  it("opens newest first when the URL names no order", () => {
    const { result } = renderRouting("?q=agent:research");
    expect(result.current[0]).toEqual(NEWEST);
  });

  it("reads the order the URL names", () => {
    const { result } = renderRouting("?sort_by=duration_ms&sort_dir=asc");
    expect(result.current[0]).toEqual({ key: "duration_ms", descending: false });
  });

  it("falls back to newest first on an order the server cannot page by", () => {
    const { result } = renderRouting("?sort_by=cost&sort_dir=sideways");
    expect(result.current[0]).toEqual(NEWEST);
  });

  it("writes a chosen order to the URL, leaving out whatever matches the default, and drops it back out", async () => {
    const { result, lastUrl } = renderRouting("");
    await act(async () => result.current[1]({ key: "error_count", descending: false }));
    expect(result.current[0]).toEqual({ key: "error_count", descending: false });
    await waitFor(() => expect(lastUrl().get("sort_by")).toBe("error_count"));
    expect(lastUrl().get("sort_dir")).toBe("asc");

    await act(async () => result.current[1]({ key: "error_count", descending: true }));
    await waitFor(() => expect(lastUrl().has("sort_dir")).toBe(false));
    expect(lastUrl().get("sort_by")).toBe("error_count");

    await act(async () => result.current[1](NEWEST));
    expect(result.current[0]).toEqual(NEWEST);
    await waitFor(() => expect(lastUrl().has("sort_by")).toBe(false));
    expect(lastUrl().has("sort_dir")).toBe(false);
  });
});
