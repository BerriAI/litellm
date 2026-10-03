"use client";

import { useCallback, useState } from "react";

const KEY = "litellm.lens.showWork";

function readStored(): boolean {
  try {
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

function store(open: boolean): void {
  try {
    window.localStorage.setItem(KEY, open ? "1" : "0");
  } catch {
    return;
  }
}

export function useShowWork(): [boolean, () => void] {
  const [open, setOpen] = useState(readStored);
  const toggle = useCallback(() => {
    setOpen((current) => {
      store(!current);
      return !current;
    });
  }, []);
  return [open, toggle];
}
