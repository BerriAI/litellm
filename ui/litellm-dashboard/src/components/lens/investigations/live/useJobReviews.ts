"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { lensQueries } from "../../api/queries";
import type { LensApi } from "../../api/service";
import { appendPage, EMPTY_FEED, type ReviewFeed } from "../../model/live";
import type { Job, Review } from "../../model/types";

export function useJobReviews(api: LensApi, lensId: string, job: Pick<Job, "id" | "status">): readonly Review[] {
  const [feed, setFeed] = useState<ReviewFeed>(EMPTY_FEED);
  const live = job.status === "queued" || job.status === "running";
  const { data } = useQuery(lensQueries.reviews(api, { lensId, jobId: job.id, after: feed.cursor, live }));
  const next = data ? appendPage(feed, data) : feed;
  if (next !== feed) setFeed(next);
  return next.reviews;
}
