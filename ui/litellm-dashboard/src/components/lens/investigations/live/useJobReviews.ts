"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { lensQueries } from "../../data/queries";
import type { LensApi } from "../../data/service";
import { appendPage, EMPTY_FEED, polling, type ReviewFeed } from "../../model/live";
import type { Job, Review } from "../../model/types";

export function useJobReviews(
  api: LensApi,
  lensId: string,
  job: Pick<Job, "id" | "status" | "reviewed" | "attempts">,
): readonly Review[] {
  const attempt = job.attempts ?? 0;
  const [state, setState] = useState<{ attempt: number; feed: ReviewFeed }>({ attempt, feed: EMPTY_FEED });
  const feed = state.attempt === attempt ? state.feed : EMPTY_FEED;
  const page = { lensId, jobId: job.id, attempt, after: feed.cursor, live: polling(job, feed) };
  const { data } = useQuery(lensQueries.reviews(api, page));
  const next = data ? appendPage(feed, data) : feed;
  if (state.attempt !== attempt || next !== state.feed) setState({ attempt, feed: next });
  return next.reviews;
}
