"use client";

import { useEffect, useReducer, useState } from "react";

import { playbackPhase, playbackReducer, startPlayback, type Phase, type Playback } from "../../model/live";
import type { Review } from "../../model/types";

const FRAME_MS = 40;

function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function useReviewPlayback(reviews: readonly Review[], live: boolean): Playback & { phase: Phase } {
  const [still] = useState(prefersReducedMotion);
  const [state, dispatch] = useReducer(playbackReducer, undefined, () =>
    playbackReducer(startPlayback(reviews, live), still ? { type: "settle" } : { type: "tick", now: Date.now() }),
  );
  const [now, setNow] = useState(Date.now);

  useEffect(() => {
    dispatch({ type: "enqueue", reviews });
    if (still) dispatch({ type: "settle" });
  }, [reviews, still]);

  const busy = state.pending.length > 0 || now - state.startedAt < state.duration;
  useEffect(() => {
    if (!busy || still) return;
    const timer = window.setInterval(() => {
      const at = Date.now();
      setNow(at);
      dispatch({ type: "tick", now: at });
    }, FRAME_MS);
    return () => window.clearInterval(timer);
  }, [busy, still]);

  const current = state.current;
  const phase =
    still || !current
      ? { span: -1, typed: current?.reasoning.length ?? 0, verdict: true }
      : playbackPhase(now - state.startedAt, state.duration, current.spans.length, current.reasoning.length);
  return { ...state, phase };
}
