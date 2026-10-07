import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../../../tests/test-utils";
import { type TracesApi, TracesApiContext } from "../../api";
import type { Feedback, TraceFeedback } from "../../types";
import { FeedbackPanel } from "./FeedbackPanel";

const summary = { trace_id: "trace-1", trace_ref: "REF1" };

const entry = (author: string, score: number, comment: string): Feedback => ({
  ...summary,
  author,
  score,
  comment,
  created_at: "2026-03-01T12:00:00Z",
  updated_at: "2026-03-01T12:00:00Z",
});

function stubApi(initial: TraceFeedback) {
  const state = { current: initial };
  const api = {
    live: true,
    feedback: vi.fn(async () => state.current),
    submitFeedback: vi.fn(async (submission: { score: number; comment?: string }) => {
      const saved = entry("me", submission.score, submission.comment ?? "");
      state.current = { ...state.current, feedback: [...state.current.feedback.filter((e) => e.author !== "me"), saved] };
      return saved;
    }),
    deleteFeedback: vi.fn(async () => {
      state.current = { ...state.current, feedback: state.current.feedback.filter((e) => e.author !== "me") };
    }),
  };
  return api;
}

const renderPanel = (api: ReturnType<typeof stubApi>) =>
  renderWithProviders(
    <TracesApiContext.Provider value={api as unknown as TracesApi}>
      <FeedbackPanel summary={summary} accessToken="sk-test" />
    </TracesApiContext.Provider>,
  );

describe("FeedbackPanel", () => {
  beforeEach(() => testQueryClient.clear());

  it("shows the average and count on the button and flags a run that someone scored low", async () => {
    renderPanel(stubApi({ ...summary, viewer: "me", feedback: [entry("alice", 2, "wrong file"), entry("bob", 8, "")] }));

    const button = await screen.findByRole("button", { name: /Feedback/ });
    expect(await within(button).findByTestId("feedback-count")).toHaveTextContent("5.0 · 2");
    expect(button).toHaveClass("text-destructive");
  });

  it("lists everyone's feedback and saves the viewer's score and comment for this trace", async () => {
    const user = userEvent.setup();
    const api = stubApi({ ...summary, viewer: "me", feedback: [entry("alice", 2, "picked the wrong file")] });
    renderPanel(api);

    await user.click(await screen.findByRole("button", { name: /Feedback/ }));
    const others = await screen.findByRole("list", { name: "Feedback from others" });
    expect(within(others).getByText("alice")).toBeInTheDocument();
    expect(within(others).getByText("picked the wrong file")).toBeInTheDocument();
    expect(within(others).getByText("2/10")).toHaveClass("text-destructive");

    const save = screen.getByRole("button", { name: "Save" });
    expect(save).toBeDisabled();
    await user.click(screen.getByRole("radio", { name: "7" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Feedback comment" }), {
      target: { value: "Right answer after one retry" },
    });
    await user.click(save);

    await waitFor(() =>
      expect(api.submitFeedback).toHaveBeenCalledWith({
        trace_id: "trace-1",
        trace_ref: "REF1",
        score: 7,
        comment: "Right answer after one retry",
      }),
    );
    expect(await screen.findByRole("button", { name: "Update" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "7" })).toHaveAttribute("aria-checked", "true");
  });

  it("lets the viewer remove their own feedback", async () => {
    const user = userEvent.setup();
    const api = stubApi({ ...summary, viewer: "me", feedback: [entry("me", 3, "too slow")] });
    renderPanel(api);

    await user.click(await screen.findByRole("button", { name: /Feedback/ }));
    expect(await screen.findByRole("textbox", { name: "Feedback comment" })).toHaveValue("too slow");
    await user.click(screen.getByRole("button", { name: "Remove" }));

    await waitFor(() => expect(api.deleteFeedback).toHaveBeenCalledWith("trace-1", "REF1"));
    expect(await screen.findByRole("button", { name: "Save" })).toBeDisabled();
    expect(screen.getByText("No feedback on this run yet.")).toBeInTheDocument();
  });
});
