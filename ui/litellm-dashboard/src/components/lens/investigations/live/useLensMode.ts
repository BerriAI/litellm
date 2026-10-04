"use client";

import { useEffect, useState } from "react";

import type { LensMode } from "./TraceList";

const SETTLE_MS = 400;

export function useLensMode(live: boolean): LensMode {
  const [wasLive, setWasLive] = useState(live);
  const [settled, setSettled] = useState(!live);
  if (live && !wasLive) {
    setWasLive(true);
    setSettled(false);
  }
  useEffect(() => {
    if (live || settled) return;
    const timer = window.setTimeout(() => setSettled(true), SETTLE_MS);
    return () => window.clearTimeout(timer);
  }, [live, settled]);
  if (live) return "live";
  return settled ? "off" : "settling";
}
