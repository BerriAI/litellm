"use client";

import { useEffect, useState } from "react";

import type { InFlight } from "../../model/live";
import { startStage, stepStage, type Stage } from "../../model/stage";
import type { Review } from "../../model/types";

const TICK_MS = 30;
const CHAR_MS = 3;

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
}

export function useStage(
  reviews: readonly Review[],
  reading: readonly InFlight[],
  running: boolean,
  slots: number,
): { stage: Stage; now: number; charMs: number } {
  const [charMs] = useState(() => (prefersReducedMotion() ? 0 : CHAR_MS));
  const [stage, setStage] = useState(() => startStage(reviews));
  const [now, setNow] = useState(Date.now);
  const input = { reading, reviews, now, slots, running, charMs };
  const next = stepStage(stage, input);
  if (next !== stage) setStage(next);
  const busy = running || next.lanes.length > 0;
  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(() => setNow(Date.now()), TICK_MS);
    return () => window.clearInterval(timer);
  }, [busy]);
  return { stage: next, now, charMs };
}
