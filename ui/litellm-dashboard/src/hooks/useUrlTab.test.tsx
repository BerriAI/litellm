import { act, renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { useUrlTab } from "./useUrlTab";

const TABS = ["chat", "compare", "compliance"] as const;
type Tab = (typeof TABS)[number];

interface RenderArgs {
  searchParams?: string;
  onUrlUpdate?: OnUrlUpdateFunction;
  key?: string;
}

const initialProps: { values: readonly Tab[] } = { values: TABS };

const renderUrlTab = ({ searchParams, onUrlUpdate, key }: RenderArgs = {}) =>
  renderHook(({ values }: { values: readonly Tab[] }) => useUrlTab(values, "chat", key), {
    initialProps,
    wrapper: ({ children }: { children: ReactNode }) => (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        {children}
      </NuqsTestingAdapter>
    ),
  });

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

describe("useUrlTab", () => {
  it("reads the active tab from the URL", () => {
    const { result } = renderUrlTab({ searchParams: "?tab=compare" });

    expect(result.current[0]).toBe("compare");
  });

  it("resolves a URL value outside the allowed tabs to the fallback and drops it from the URL", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUrlTab({ searchParams: "?tab=settings&other=1", onUrlUpdate });

    expect(result.current[0]).toBe("chat");
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("other")).toBe("1");
  });

  it("leaves a URL that names an allowed tab untouched", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderUrlTab({ searchParams: "?tab=compare", onUrlUpdate });

    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  it("reads from the caller's key instead of the default one", () => {
    const { result } = renderUrlTab({ searchParams: "?view=compliance&tab=compare", key: "view" });

    expect(result.current[0]).toBe("compliance");
  });

  it("writes ?tab= with history replace when a tab is selected", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUrlTab({ onUrlUpdate });

    act(() => result.current[1]("compare"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("compare"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
    expect(result.current[0]).toBe("compare");
  });

  it("removes the param when the fallback tab is selected", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUrlTab({ searchParams: "?tab=compare", onUrlUpdate });

    act(() => result.current[1]("chat"));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
    expect(result.current[0]).toBe("chat");
  });

  it("falls back and clears the param when the current tab is no longer among the allowed values", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result, rerender } = renderUrlTab({ searchParams: "?tab=compliance", onUrlUpdate });
    expect(result.current[0]).toBe("compliance");

    rerender({ values: ["chat", "compare"] });

    expect(result.current[0]).toBe("chat");
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
  });
});
