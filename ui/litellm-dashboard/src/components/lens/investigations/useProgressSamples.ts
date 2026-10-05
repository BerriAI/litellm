"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { analysisFraction, type ProgressSample } from "../model/progress";

const KEEP = 120;

type Progress = Pick<ProgressSample, "step" | "done"> & { readonly total: number };

/** A wall-clock log of progress changes; the clock is the external system, so it lives outside render. */
function createSampleLog() {
  const listeners = new Set<() => void>();
  const state = { samples: [] as readonly ProgressSample[] };
  return {
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => void listeners.delete(listener);
    },
    snapshot: () => state.samples,
    record: (progress: Progress) => {
      const sample = { at: Date.now(), ...progress, fraction: analysisFraction(progress) };
      state.samples = [...state.samples, sample].slice(-KEEP);
      listeners.forEach((listener) => listener());
    },
  };
}

export function useProgressSamples({ step, done, total }: Progress): readonly ProgressSample[] {
  const [log] = useState(createSampleLog);
  useEffect(() => log.record({ step, done, total }), [log, step, done, total]);
  return useSyncExternalStore(log.subscribe, log.snapshot, log.snapshot);
}
