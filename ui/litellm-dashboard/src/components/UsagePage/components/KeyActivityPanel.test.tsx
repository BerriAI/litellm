import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

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

describe("KeyActivityPanel", () => {
  it("renders every key and the full count before searching", () => {
    render(<KeyActivityPanel keyMetrics={keyMetrics} />);
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-alicehash-bob");
    expect(screen.getByText("Showing 2 of 2 keys")).toBeInTheDocument();
  });

  it("narrows the rendered keys to those matching the user email", () => {
    render(<KeyActivityPanel keyMetrics={keyMetrics} />);
    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "bob@example.com" } });
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-bob");
    expect(screen.getByTestId("rendered-keys")).not.toHaveTextContent("hash-alice");
    expect(screen.getByText("Showing 1 of 2 keys")).toBeInTheDocument();
  });

  it("shows an empty state instead of zeroed metrics when nothing matches", () => {
    render(<KeyActivityPanel keyMetrics={keyMetrics} />);
    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "carol" } });
    expect(screen.queryByTestId("rendered-keys")).not.toBeInTheDocument();
    expect(screen.getByText('No keys match "carol" in this date range')).toBeInTheDocument();
  });

  it("clears the search and restores every key", () => {
    render(<KeyActivityPanel keyMetrics={keyMetrics} />);
    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "user-alice" } });
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-alice");
    fireEvent.click(screen.getByLabelText("Clear key search"));
    expect(screen.getByLabelText("Search keys")).toHaveValue("");
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-alicehash-bob");
  });

  it("counts toward the server-side key total when only the top spenders were loaded", () => {
    render(<KeyActivityPanel keyMetrics={keyMetrics} apiKeyTruncation={{ limit: 2, total: 3000 }} />);
    expect(screen.getByText("Showing 2 of 3,000 keys")).toBeInTheDocument();
  });

  it("shows the full server total once every key is loaded", () => {
    render(<KeyActivityPanel keyMetrics={keyMetrics} apiKeyTruncation={{ limit: 2, total: 2 }} />);
    expect(screen.getByText("Showing 2 of 2 keys")).toBeInTheDocument();
  });

  it("finds keys outside the loaded top-spend subset via server search", async () => {
    const searchKeys = vi
      .fn<(query: string) => Promise<Record<string, ModelActivityData>>>()
      .mockResolvedValue({ "hash-gamma": activity("gamma-low-key", "gamma@example.com", "user-gamma") });
    render(
      <KeyActivityPanel keyMetrics={keyMetrics} apiKeyTruncation={{ limit: 2, total: 3 }} searchKeys={searchKeys} />,
    );

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "gamma" } });

    expect(await screen.findByText("hash-gamma")).toBeInTheDocument();
    expect(searchKeys).toHaveBeenCalledWith("gamma");
    expect(screen.getByText("Showing 1 of 3 keys")).toBeInTheDocument();
  });

  it("never calls the server search when every key is already loaded", async () => {
    const searchKeys = vi
      .fn<(query: string) => Promise<Record<string, ModelActivityData>>>()
      .mockResolvedValue({ "hash-gamma": activity("gamma-low-key", "gamma@example.com", "user-gamma") });
    render(<KeyActivityPanel keyMetrics={keyMetrics} searchKeys={searchKeys} />);

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "gamma" } });

    expect(await screen.findByText('No keys match "gamma" in this date range')).toBeInTheDocument();
    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(searchKeys).not.toHaveBeenCalled();
  });

  it("drops stale server results as soon as the search callback is rebuilt", async () => {
    const searchKeysA = vi
      .fn<(query: string) => Promise<Record<string, ModelActivityData>>>()
      .mockResolvedValue({ "hash-gamma": activity("gamma-low-key", "gamma@example.com", "user-gamma") });
    const searchKeysB = vi
      .fn<(query: string) => Promise<Record<string, ModelActivityData>>>()
      .mockReturnValue(new Promise(() => {}));
    const { rerender } = render(
      <KeyActivityPanel keyMetrics={keyMetrics} apiKeyTruncation={{ limit: 2, total: 3 }} searchKeys={searchKeysA} />,
    );

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "gamma" } });
    expect(await screen.findByText("hash-gamma")).toBeInTheDocument();

    rerender(
      <KeyActivityPanel keyMetrics={keyMetrics} apiKeyTruncation={{ limit: 2, total: 3 }} searchKeys={searchKeysB} />,
    );

    expect(screen.getByRole("status")).toHaveTextContent("Searching all keys");
    expect(screen.queryByText("hash-gamma")).not.toBeInTheDocument();
  });

  it("reports a failed server search but keeps the local matches", async () => {
    const searchKeys = vi
      .fn<(query: string) => Promise<Record<string, ModelActivityData>>>()
      .mockRejectedValue(new Error("boom"));
    render(
      <KeyActivityPanel keyMetrics={keyMetrics} apiKeyTruncation={{ limit: 2, total: 3 }} searchKeys={searchKeys} />,
    );

    fireEvent.change(screen.getByLabelText("Search keys"), { target: { value: "alice" } });

    expect(await screen.findByRole("alert")).toHaveTextContent("Key search failed");
    expect(screen.getByTestId("rendered-keys")).toHaveTextContent("hash-alice");
  });

  it("shows no Load more keys button without the prop", () => {
    render(<KeyActivityPanel keyMetrics={keyMetrics} apiKeyTruncation={{ limit: 2, total: 3 }} />);
    expect(screen.queryByRole("button", { name: "Load more keys" })).not.toBeInTheDocument();
  });

  it("calls loadMoreKeys when the button is clicked", async () => {
    const loadMoreKeys = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
    render(
      <KeyActivityPanel
        keyMetrics={keyMetrics}
        apiKeyTruncation={{ limit: 2, total: 3 }}
        loadMoreKeys={loadMoreKeys}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Load more keys" }));

    expect(loadMoreKeys).toHaveBeenCalledTimes(1);
    expect(await screen.findByRole("button", { name: "Load more keys" })).toBeEnabled();
  });

  it("shows a pending status while loadMoreKeys is in flight", async () => {
    const loadMoreKeys = vi.fn<() => Promise<void>>().mockReturnValue(new Promise(() => {}));
    render(
      <KeyActivityPanel
        keyMetrics={keyMetrics}
        apiKeyTruncation={{ limit: 2, total: 3 }}
        loadMoreKeys={loadMoreKeys}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Load more keys" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Loading more keys...");
    expect(screen.getByRole("button", { name: "Load more keys" })).toBeDisabled();
  });

  it("reports a failed key page load", async () => {
    const loadMoreKeys = vi.fn<() => Promise<void>>().mockRejectedValue(new Error("boom"));
    render(
      <KeyActivityPanel
        keyMetrics={keyMetrics}
        apiKeyTruncation={{ limit: 2, total: 3 }}
        loadMoreKeys={loadMoreKeys}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Load more keys" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Loading more keys failed");
  });

  it("clears a stale load-more status when the page callback changes", async () => {
    const staleLoadMoreKeys = vi.fn<() => Promise<void>>().mockRejectedValue(new Error("boom"));
    const { rerender } = render(
      <KeyActivityPanel
        keyMetrics={keyMetrics}
        apiKeyTruncation={{ limit: 2, total: 3 }}
        loadMoreKeys={staleLoadMoreKeys}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Load more keys" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Loading more keys failed");

    const freshLoadMoreKeys = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
    rerender(
      <KeyActivityPanel
        keyMetrics={keyMetrics}
        apiKeyTruncation={{ limit: 2, total: 3 }}
        loadMoreKeys={freshLoadMoreKeys}
      />,
    );

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
