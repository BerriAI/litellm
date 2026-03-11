"use client";
// hooks/useDisableShowPrompts.ts
import { useSyncExternalStore } from "react";
import { getLocalStorageItem } from "@/utils/localStorageUtils";
import { LOCAL_STORAGE_EVENT } from "@/utils/localStorageUtils";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

function subscribe(callback: () => void) {
  const onStorage = (e: StorageEvent) => {
    if (e.key === "disableShowPrompts") {
      callback();
    }
  };

  const onCustom = (e: Event) => {
    const { key } = (e as CustomEvent).detail;
    if (key === "disableShowPrompts") {
      callback();
    }
  };

  window.addEventListener("storage", onStorage);
  window.addEventListener(LOCAL_STORAGE_EVENT, onCustom);

  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(LOCAL_STORAGE_EVENT, onCustom);
  };
}

function getSnapshot(): string | null {
  return getLocalStorageItem("disableShowPrompts");
}

function getServerSnapshot(): string | null {
  return null;
}

export function useDisableShowPrompts(): boolean {
  const { premiumUser } = useAuthorized();
  const storedRaw = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  // Non-premium users always see prompts regardless of stored preference
  if (!premiumUser) return false;

  // Premium users: absent (null) or explicitly "true" means hidden; "false" means explicitly shown
  return storedRaw === "true" || storedRaw === null;
}
