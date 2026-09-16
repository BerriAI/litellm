import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";

import type { ModelActivityData } from "../types";
import KeyActivityPanel from "./KeyActivityPanel";

vi.mock("@/components/activity_metrics", () => ({
  ActivityMetrics: ({ modelMetrics }: { modelMetrics: Record<string, ModelActivityData> }) => (
    <ul data-testid="rendered-keys">
      {Object.keys(modelMetrics).map((hash) => (
        <li key={hash}>{hash}</li>
      ))}
    </ul>
  ),
}));

function activity(label: string, user_email: string | null, user_id: string | null): ModelActivityData {
  return {
    label,
    key_metadata: { key_alias: label, team_id: "team-1", user_id, user_email },
    total_requests: 1,
    total_successful_requests: 1,
    total_failed_requests: 0,
    total_cache_read_input_tokens: 0,
    total_cache_creation_input_tokens: 0,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    total_spend: 0.01,
    top_api_keys: [],
    top_models: [],
    daily_data: [],
  };
}

const keyMetrics: Record<string, ModelActivityData> = {
  "hash-alice": activity("alice-key", "alice@example.com", "user-alice"),
  "hash-bob": activity("bob-key", "bob@example.com", "user-bob"),
};

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

describe("KeyActivityPanel", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders every key and the full count before searching", () => {
    renderWithProviders(<KeyActivityPanel keyMetrics={keyMetrics} />);
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-alicehash-bob");
    expect(screen.getByText("Showing 2 of 2 keys")).toBeInTheDocument();
  });

  it("narrows the rendered keys to those matching the user email", () => {
    renderWithProviders(<KeyActivityPanel keyMetrics={keyMetrics} />);
    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "bob@example.com" } });
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-bob");
    expect(screen.getByTestId("rendered-keys")).not.toHaveTextContent("hash-alice");
    expect(screen.getByText("Showing 1 of 2 keys")).toBeInTheDocument();
  });

  it("shows an empty state instead of zeroed metrics when nothing matches", () => {
    renderWithProviders(<KeyActivityPanel keyMetrics={keyMetrics} />);
    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "carol" } });
    expect(screen.queryByTestId("rendered-keys")).not.toBeInTheDocument();
    expect(screen.getByText('No keys match "carol" in this date range')).toBeInTheDocument();
  });

  it("clears the search and restores every key", () => {
    renderWithProviders(<KeyActivityPanel keyMetrics={keyMetrics} />);
    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "user-alice" } });
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-alice");
    fireEvent.click(screen.getByLabelText("Clear key search"));
    expect(screen.getByLabelText("Search keys")).toHaveValue("");
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-alicehash-bob");
  });

  it("applies the search from ?key_search= on load", () => {
    renderWithProviders(<KeyActivityPanel keyMetrics={keyMetrics} />, { searchParams: "?key_search=bob" });
    expect(screen.getByLabelText("Search keys")).toHaveValue("bob");
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-bob");
    expect(screen.getByTestId("rendered-keys")).not.toHaveTextContent("hash-alice");
    expect(screen.getByText("Showing 1 of 2 keys")).toBeInTheDocument();
  });

  it("writes ?key_search= while typing and drops it when the search is cleared", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<KeyActivityPanel keyMetrics={keyMetrics} />, { onUrlUpdate });

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "user-alice" } });
    await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("key_search")).toBe("user-alice"));
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].options.history).toBe("replace");

    fireEvent.click(screen.getByLabelText("Clear key search"));
    await waitFor(() => expect(lastUrl(onUrlUpdate)?.has("key_search")).toBe(false));
  });

  it("writes ?key_search= at most once every 300ms while typing", async () => {
    vi.useFakeTimers();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(<KeyActivityPanel keyMetrics={keyMetrics} />, {
      wrapper: ({ children }: { children: ReactNode }) => (
        <NuqsTestingAdapter onUrlUpdate={onUrlUpdate} hasMemory rateLimitFactor={1} resetUrlUpdateQueueOnMount={false}>
          {children}
        </NuqsTestingAdapter>
      ),
    });
    const input = screen.getByLabelText("Search keys");

    fireEvent.change(input, { target: { value: "a" } });
    await act(() => vi.runAllTimersAsync());
    expect(onUrlUpdate).toHaveBeenCalledTimes(1);

    fireEvent.change(input, { target: { value: "bob" } });
    await act(() => vi.advanceTimersByTimeAsync(100));
    expect(onUrlUpdate).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-bob");
    expect(screen.getByTestId("rendered-keys")).not.toHaveTextContent("hash-alice");

    await act(() => vi.advanceTimersByTimeAsync(250));
    expect(onUrlUpdate).toHaveBeenCalledTimes(2);
    expect(lastUrl(onUrlUpdate)?.get("key_search")).toBe("bob");
  });
});
