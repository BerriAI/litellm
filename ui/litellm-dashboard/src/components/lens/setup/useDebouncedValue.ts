"use client";

import { hashKey } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

interface Settled<T> {
  readonly value: T;
  readonly key: string;
  readonly at: string;
}

export interface DebouncedValue<T> {
  readonly value: T;
  readonly pending: boolean;
  readonly settledAt: string;
}

/** Debounces by structure, so a value that only changed by reference neither restarts the wait nor settles again. */
export function useDebouncedValue<T>(value: T, wait: number): DebouncedValue<T> {
  const key = hashKey([value]);
  const [settled, setSettled] = useState<Settled<T>>(() => ({ value, key, at: new Date().toISOString() }));
  const latest = useRef(value);
  const pending = key !== settled.key;
  useEffect(() => {
    latest.current = value;
  });
  useEffect(() => {
    if (!pending) return;
    const timer = window.setTimeout(
      () => setSettled({ value: latest.current, key, at: new Date().toISOString() }),
      wait,
    );
    return () => window.clearTimeout(timer);
  }, [key, pending, wait]);
  return { value: settled.value, pending, settledAt: settled.at };
}
