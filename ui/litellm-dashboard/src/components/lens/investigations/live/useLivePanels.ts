"use client";

import { useCallback, useState } from "react";

const STRIP_KEY = "litellm.lens.liveStrip";

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    return;
  }
}

export function useStripOpen(): [boolean, (open: boolean) => void] {
  const [open, setOpen] = useState(() => read(STRIP_KEY) !== "closed");
  const update = useCallback((next: boolean) => {
    setOpen(next);
    write(STRIP_KEY, next ? "open" : "closed");
  }, []);
  return [open, update];
}
