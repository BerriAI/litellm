import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import { createLensDemo } from "../../data/demo/createLensDemo";
import { createLensDemoData } from "../../data/demo/fixtures";
import type { LensApi } from "../../data/service";
import type { Job } from "../../model/types";
import { useJobReviews } from "./useJobReviews";

function Reviews({ api, job }: { api: LensApi; job: Job }) {
  const reviews = useJobReviews(api, "support", job);
  return (
    <ul>
      {reviews.map((review) => (
        <li key={review.execution_id}>{review.reasoning}</li>
      ))}
    </ul>
  );
}

it("replaces cached reviews when a worker reclaims the same run", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const job = createLensDemoData().lenses[0].jobs[0];
  const previous = { ...job.reviews[0], reasoning: "Previous attempt's conclusion" };
  const current = { ...job.reviews[0], reasoning: "Current attempt's conclusion", at: "2026-10-05T12:00:00Z" };
  const reviews = vi.fn<LensApi["reviews"]>().mockImplementation(async (_lens, _job, after) => ({
    reviews: after === 0 ? [previous] : [],
    reviewed: 1,
  }));
  const api = { ...createLensDemo().lens, reviews };
  const view = (attempts: number) => (
    <QueryClientProvider client={client}>
      <Reviews api={api} job={{ ...job, attempts, reviewed: 1 }} />
    </QueryClientProvider>
  );
  const { rerender } = render(view(1));
  expect(await screen.findByText(previous.reasoning)).toBeVisible();
  reviews.mockImplementation(async (_lens, _job, after) => ({
    reviews: after === 0 ? [current] : [],
    reviewed: 1,
  }));
  rerender(view(2));
  expect(await screen.findByText(current.reasoning)).toBeVisible();
  expect(screen.queryByText(previous.reasoning)).not.toBeInTheDocument();
});
