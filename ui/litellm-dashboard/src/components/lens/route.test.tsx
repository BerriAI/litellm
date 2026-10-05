import { act, renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { PropsWithChildren } from "react";
import { describe, expect, it, vi } from "vitest";

import { useDemoRoute, useDialogRoute, useInvestigateRoute } from "./route";

const renderRoute = <T,>(hook: () => T, searchParams: string) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  const wrapper = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter searchParams={searchParams} onUrlUpdate={onUrlUpdate} hasMemory>
      {children}
    </NuqsTestingAdapter>
  );
  const view = renderHook(hook, { wrapper });
  const lastUrl = () => new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString ?? "");
  return { ...view, lastUrl };
};

describe("setup routes", () => {
  it("opens a new investigation on the searched runs, dropping any older draft", async () => {
    const { result, lastUrl } = renderRoute(useInvestigateRoute, "?tab=traces&q=old&context=stale&step=run");
    await act(async () => result.current({ q: "agent:support status:error", lookbackHours: 168 }));
    await waitFor(() => expect(lastUrl().get("dialog")).toBe("new"));
    const expected = { tab: "investigations", dialog: "new", q: "agent:support status:error", lookback: "168" };
    expect(Object.fromEntries(lastUrl())).toEqual(expected);
  });

  it("closing setup clears the draft but keeps the run search the Traces tab shares", async () => {
    const { result, lastUrl } = renderRoute(
      useDialogRoute,
      "?tab=investigations&dialog=new&q=agent:support&name=Refunds&lookback=72&watch=watch_unsafe&step=criteria",
    );
    await act(async () => result.current.closeDialog());
    await waitFor(() => expect(lastUrl().has("dialog")).toBe(false));
    expect(Object.fromEntries(lastUrl())).toEqual({ tab: "investigations", q: "agent:support" });
  });
});

describe("demo route", () => {
  const live = "?tab=investigations&lens=l1&run=b1&finding=f1&q=agent:support&dialog=new&name=Refunds";

  it("entering the sample session keeps only the tab so live ids never leak into it", async () => {
    const { result, lastUrl } = renderRoute(useDemoRoute, live);
    await act(async () => result.current(true));
    await waitFor(() => expect(lastUrl().get("demo")).toBe("true"));
    expect(Object.fromEntries(lastUrl())).toEqual({ tab: "investigations", demo: "true" });
  });

  it("leaving the sample session drops its ids and keeps the tab", async () => {
    const { result, lastUrl } = renderRoute(useDemoRoute, "?tab=traces&demo=true&lens=sample&q=agent:demo");
    await act(async () => result.current(false));
    await waitFor(() => expect(lastUrl().has("demo")).toBe(false));
    expect(Object.fromEntries(lastUrl())).toEqual({ tab: "traces" });
  });
});
