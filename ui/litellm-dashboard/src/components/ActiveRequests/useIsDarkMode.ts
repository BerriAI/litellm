"use client";

import { useSyncExternalStore } from "react";

const subscribe = (onStoreChange: () => void): (() => void) => {
  const observer = new MutationObserver(onStoreChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  return () => observer.disconnect();
};

const getSnapshot = (): boolean => document.documentElement.classList.contains("dark");

const getServerSnapshot = (): boolean => false;

export const useIsDarkMode = (): boolean => useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
