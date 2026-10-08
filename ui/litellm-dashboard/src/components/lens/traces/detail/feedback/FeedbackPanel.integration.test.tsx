import { screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../../../tests/test-utils";
import { type TracesApi, TracesApiContext } from "../../api";
import type { Feedback } from "../../types";
import { FeedbackPanel } from "./FeedbackPanel";

const summary = { trace_id: "trace-1", trace_ref: "REF1" };

const entry = (author: string, score: number, comment: string, updated_at = "2026-03-01T12:00:00Z"): Feedback => ({
  ...summary,
  author,
  score,
  comment,
  created_at: updated_at,
  updated_at,
});

const renderPanel = (feedback: Feedback[]) => {
  const api = { live: true, feedback: vi.fn(async () => ({ ...summary, feedback })) };
  renderWithProviders(
    <TracesApiContext.Provider value={api as unknown as TracesApi}>
      <FeedbackPanel summary={summary} accessToken="sk-test" />
    </TracesApiContext.Provider>,
  );
  return api;
};

describe("FeedbackPanel", () => {
  beforeEach(() => testQueryClient.clear());

  it("shows what the end user said and their score, flagged when it is low", async () => {
    const api = renderPanel([entry("customer-1042", 2, "It edited the wrong file and I had to ask twice")]);

    const panel = await screen.findByRole("region", { name: "User feedback" });
    expect(panel).toHaveAttribute("data-low", "true");
    const row = within(panel).getByTestId("feedback-entry");
    expect(row).toHaveTextContent("2/10");
    expect(row).toHaveTextContent("“It edited the wrong file and I had to ask twice”");
    expect(row).toHaveTextContent("customer-1042");
    expect(api.feedback).toHaveBeenCalledWith("trace-1", "REF1");
  });

  it("lists every user's feedback newest first with the average when several users rated the run", async () => {
    renderPanel([
      entry("customer-1", 9, "Perfect", "2026-03-01T12:00:00Z"),
      entry("customer-2", 3, "Too slow", "2026-03-01T12:05:00Z"),
    ]);

    const panel = await screen.findByRole("region", { name: "User feedback" });
    expect(
      within(panel)
        .getAllByTestId("feedback-entry")
        .map((row) => row.textContent),
    ).toEqual([expect.stringContaining("customer-2"), expect.stringContaining("customer-1")]);
    expect(panel).toHaveTextContent("6/10 avg from 2 users");
    expect(panel).toHaveAttribute("data-low", "true");
  });

  it("does not flag a run every user scored well and offers no way to edit feedback", async () => {
    renderPanel([entry("customer-1", 8, "")]);

    const panel = await screen.findByRole("region", { name: "User feedback" });
    expect(panel).not.toHaveAttribute("data-low");
    expect(within(panel).getByText("No comment")).toBeInTheDocument();
    expect(within(panel).queryByRole("button")).not.toBeInTheDocument();
    expect(within(panel).queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("stays out of the way when feedback cannot be loaded so the run still reads cleanly", async () => {
    const api = { live: true, feedback: vi.fn(() => Promise.reject(new Error("ClickHouse down"))) };
    renderWithProviders(
      <TracesApiContext.Provider value={api as unknown as TracesApi}>
        <FeedbackPanel summary={summary} accessToken="sk-test" />
      </TracesApiContext.Provider>,
    );
    await vi.waitFor(() => expect(api.feedback).toHaveBeenCalled());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "User feedback" })).not.toBeInTheDocument();
  });

  it("renders nothing when no end user rated the run", async () => {
    const api = renderPanel([]);
    await vi.waitFor(() => expect(api.feedback).toHaveBeenCalled());
    expect(screen.queryByRole("region", { name: "User feedback" })).not.toBeInTheDocument();
  });
});
