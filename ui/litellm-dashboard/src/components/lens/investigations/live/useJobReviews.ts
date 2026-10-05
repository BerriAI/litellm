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
  job: Pick<Job, "id" | "status" | "reviewed">,
): readonly Review[] {
  const [feed, setFeed] = useState<ReviewFeed>(EMPTY_FEED);
  const page = { lensId, jobId: job.id, after: feed.cursor, live: polling(job, feed) };
  const { data } = useQuery(lensQueries.reviews(api, page));
  const next = data ? appendPage(feed, data) : feed;
  if (next !== feed) setFeed(next);
  return next.reviews;
}
