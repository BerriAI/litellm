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
});
