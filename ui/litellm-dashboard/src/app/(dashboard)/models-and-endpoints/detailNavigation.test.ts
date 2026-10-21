import { act, renderHook, waitFor } from "@testing-library/react";
import { withNuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { describe, expect, it, vi } from "vitest";
import { useModelDetailRouting, useModelGroupFilterRouting } from "./detailNavigation";

describe("useModelDetailRouting", () => {
  it("openModel sets ?model= with a history push", async () => {
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    const { result } = renderHook(() => useModelDetailRouting(), {
      wrapper: withNuqsTestingAdapter({ onUrlUpdate }),
    });
    await act(async () => {
      result.current.openModel("abc-1");
    });
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const event = onUrlUpdate.mock.calls.at(-1)?.[0];
    expect(event?.searchParams.get("model")).toBe("abc-1");
    expect(event?.options.history).toBe("push");
  });

  it("openTeam sets ?team= and drops any model param", async () => {
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    const { result } = renderHook(() => useModelDetailRouting(), {
      wrapper: withNuqsTestingAdapter({ searchParams: "?model=abc-1", onUrlUpdate }),
    });
    await act(async () => {
      result.current.openTeam("team-9");
    });
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const event = onUrlUpdate.mock.calls.at(-1)?.[0];
    expect(event?.searchParams.get("team")).toBe("team-9");
    expect(event?.searchParams.has("model")).toBe(false);
  });

  it("close removes both model and team params", async () => {
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    const { result } = renderHook(() => useModelDetailRouting(), {
      wrapper: withNuqsTestingAdapter({ searchParams: "?model=abc-1&team=team-9", onUrlUpdate }),
    });
    await act(async () => {
      result.current.close();
    });
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const event = onUrlUpdate.mock.calls.at(-1)?.[0];
    expect(event?.searchParams.has("model")).toBe(false);
    expect(event?.searchParams.has("team")).toBe(false);
  });

  it("reads modelId and teamId from the query string", () => {
    const { result } = renderHook(() => useModelDetailRouting(), {
      wrapper: withNuqsTestingAdapter({ searchParams: "?model=xyz" }),
    });
    expect(result.current.modelId).toBe("xyz");
    expect(result.current.teamId).toBeNull();
  });
});

describe("useModelGroupFilterRouting", () => {
  it("reads the selected group from ?model_group=", () => {
    const { result } = renderHook(() => useModelGroupFilterRouting(), {
      wrapper: withNuqsTestingAdapter({ searchParams: "?model_group=gpt-4.1" }),
    });
    expect(result.current.modelGroup).toBe("gpt-4.1");
  });

  it("writes the selected group to ?model_group= and clears it on null", async () => {
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    const { result } = renderHook(() => useModelGroupFilterRouting(), {
      wrapper: withNuqsTestingAdapter({ onUrlUpdate }),
    });
    await act(async () => {
      result.current.setModelGroup("claude-sonnet-5");
    });
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("model_group")).toBe("claude-sonnet-5");

    await act(async () => {
      result.current.setModelGroup(null);
    });
    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("model_group")).toBe(false));
  });
});
