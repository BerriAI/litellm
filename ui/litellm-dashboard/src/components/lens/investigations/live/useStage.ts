"use client";

import { useEffect, useState } from "react";

import { stageTick, startStage, stepStage, type Stage } from "../../model/stage";
import type { InFlight, Review } from "../../model/types";

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
  const tick = stageTick(next, running, now, charMs);
  useEffect(() => {
    if (!tick) return;
    const timer = window.setInterval(() => setNow(Date.now()), tick);
    return () => window.clearInterval(timer);
  }, [tick]);
  return { stage: next, now, charMs };
}
