import { act, renderHook, waitFor } from "@testing-library/react";
import moment from "moment";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { useUsageUrlState } from "./useUsageUrlState";

type UrlUpdateSpy = ReturnType<typeof vi.fn<OnUrlUpdateFunction>>;

const renderUsageUrlState = (searchParams = "", onUrlUpdate: UrlUpdateSpy = vi.fn<OnUrlUpdateFunction>()) =>
  renderHook(() => useUsageUrlState(), {
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

const lastUrl = (onUrlUpdate: UrlUpdateSpy) => new URLSearchParams(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams);

describe("useUsageUrlState", () => {
  it("falls back to the global view over the last seven days with nothing in the URL", () => {
    const { result } = renderUsageUrlState();

    expect(result.current.view).toBe("global");
    expect(result.current.user).toBeNull();
    expect(result.current.modelView).toBe("groups");
    expect(result.current.topKeys).toBe(5);
    expect(result.current.topModels).toBe(5);
    expect(result.current.topAgents).toBe(5);
    expect(result.current.filter).toEqual([]);
    expect(moment(result.current.dateValue.to).diff(result.current.dateValue.from, "days")).toBe(7);
    expect(result.current.dateValue.to!.getTime()).toBeLessThanOrEqual(Date.now());
  });

  it("reads ?from= and ?to= as whole local days", () => {
    const { result } = renderUsageUrlState("?from=2025-03-01&to=2025-03-05");

    expect(result.current.dateValue.from).toEqual(new Date(2025, 2, 1));
    expect(result.current.dateValue.to).toEqual(new Date(2025, 2, 5, 23, 59, 59, 999));
  });

  it("reads every other key from the URL", () => {
    const { result } = renderUsageUrlState(
      "?view=team&user=user-9&model_view=individual&top_keys=10&top_models=25&top_agents=50&filter=team-1,team-2",
    );

    expect(result.current.view).toBe("team");
    expect(result.current.user).toBe("user-9");
    expect(result.current.modelView).toBe("individual");
    expect(result.current.topKeys).toBe(10);
    expect(result.current.topModels).toBe(25);
    expect(result.current.topAgents).toBe(50);
    expect(result.current.filter).toEqual(["team-1", "team-2"]);
  });

  it("ignores URL values outside what the page offers", () => {
    const { result } = renderUsageUrlState("?view=nope&from=03/01/2025&model_view=flat&top_keys=7&top_models=abc");

    expect(result.current.view).toBe("global");
    expect(result.current.modelView).toBe("groups");
    expect(result.current.topKeys).toBe(5);
    expect(result.current.topModels).toBe(5);
    expect(moment(result.current.dateValue.to).diff(result.current.dateValue.from, "days")).toBe(7);
  });

  it("writes a picked range to the URL as calendar days in a single update", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUsageUrlState("", onUrlUpdate);

    act(() => {
      result.current.setDateValue({ from: new Date(2024, 0, 1, 9, 30), to: new Date(2024, 0, 8, 17, 45) });
    });

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledTimes(1));
    expect(lastUrl(onUrlUpdate).get("from")).toBe("2024-01-01");
    expect(lastUrl(onUrlUpdate).get("to")).toBe("2024-01-08");
    expect(result.current.dateValue.from).toEqual(new Date(2024, 0, 1));
    expect(result.current.dateValue.to).toEqual(new Date(2024, 0, 8, 23, 59, 59, 999));
  });

  it("drops the entity filter only when the view actually changes", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUsageUrlState("?view=team&filter=team-1,team-2", onUrlUpdate);

    act(() => {
      result.current.setView("team");
    });
    expect(result.current.filter).toEqual(["team-1", "team-2"]);

    act(() => {
      result.current.setView("tag");
    });
    expect(result.current.view).toBe("tag");
    expect(result.current.filter).toEqual([]);
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("view")).toBe("tag"));
    expect(lastUrl(onUrlUpdate).has("filter")).toBe(false);
  });

  it("round-trips the entity filter as a comma separated list", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUsageUrlState("", onUrlUpdate);

    act(() => {
      result.current.setFilter(["org-1", "org-2"]);
    });

    expect(result.current.filter).toEqual(["org-1", "org-2"]);
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("filter")).toBe("org-1,org-2"));

    act(() => {
      result.current.setFilter([]);
    });
    await waitFor(() => expect(lastUrl(onUrlUpdate).has("filter")).toBe(false));
  });

  it("removes a top limit from the URL when handed a size the page does not offer", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUsageUrlState("?top_keys=25", onUrlUpdate);

    act(() => {
      result.current.setTopKeys(7);
    });

    expect(result.current.topKeys).toBe(5);
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrl(onUrlUpdate).has("top_keys")).toBe(false);
  });

  it("writes the user, model view and top limits under their own keys", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { result } = renderUsageUrlState("", onUrlUpdate);

    act(() => {
      result.current.setUser("user-1");
      result.current.setModelView("individual");
      result.current.setTopKeys(10);
      result.current.setTopModels(25);
      result.current.setTopAgents(50);
    });

    await waitFor(() => expect(lastUrl(onUrlUpdate).get("top_agents")).toBe("50"));
    const url = lastUrl(onUrlUpdate);
    expect(url.get("user")).toBe("user-1");
    expect(url.get("model_view")).toBe("individual");
    expect(url.get("top_keys")).toBe("10");
    expect(url.get("top_models")).toBe("25");

    act(() => {
      result.current.setUser(null);
      result.current.setModelView("groups");
    });
    await waitFor(() => expect(lastUrl(onUrlUpdate).has("user")).toBe(false));
    expect(lastUrl(onUrlUpdate).has("model_view")).toBe(false);
  });
});
