"use client";

import { useState } from "react";

import { modelsUsed } from "../../model/inbox";
import {
  analysisModel,
  conclusions,
  focusedReview,
  issueCount,
  reviewKey,
  shownCount,
  stripState,
} from "../../model/live";
import type { Job, Review } from "../../model/types";
import { LiveDrawer } from "./LiveDrawer";
import { LiveStrip } from "./LiveStrip";
import { useDrawerOpen, useStripOpen } from "./useLivePanels";
import { useReviewPlayback } from "./useReviewPlayback";

export function LiveRun({ job, name }: { job: Job; name: string }) {
  const live = job.status === "queued" || job.status === "running";
  const [stripOpen, setStripOpen] = useStripOpen();
  const [drawerOpen, setDrawerOpen] = useDrawerOpen(job.id, live);
  const [pinned, setPinned] = useState<string | null>(null);
  const playback = useReviewPlayback(job.reviews, live);
  const model = analysisModel([...job.reviews.map((r) => r.model), ...modelsUsed(job.steps), job.settings.model]);
  const { review: focused, following } = focusedReview(job.reviews, pinned, playback.current);
  const issues = issueCount(job);
  const reviewed = live ? shownCount(job.reviewed, playback) : job.reviewed;
  const open = (review?: Review) => {
    setPinned(review && review !== playback.current ? reviewKey(review) : null);
    setDrawerOpen(true);
  };

  return (
    <>
      {stripOpen && (
        <LiveStrip
          model={model}
          state={stripState(job, model)}
          playback={playback}
          reviewed={reviewed}
          selected={job.coverage.selected}
          issues={issues}
          cost={job.cost}
          onOpen={open}
          onClose={() => setStripOpen(false)}
        />
      )}
      {!stripOpen && (
        <button
          type="button"
          onClick={() => setStripOpen(true)}
          className="self-start text-[11px] text-muted-foreground hover:text-foreground"
        >
          Show live trace results
        </button>
      )}
      <LiveDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        name={name}
        model={model}
        job={job}
        playback={playback}
        live={live}
        focused={focused}
        following={following}
        phase={playback.phase}
        groups={conclusions(job.reviews, job.settings.checks)}
        scope={issues.scope === "findings" ? `from the last ${job.reviews.length} reviews` : issues.scope}
        onPick={(review) => setPinned(review === playback.current ? null : reviewKey(review))}
        onFollow={() => setPinned(null)}
      />
    </>
  );
}
