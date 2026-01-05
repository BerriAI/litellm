// hooks/useDisableShowNewBadge.ts
import { useSyncExternalStore } from "react";
import { getLocalStorageItem } from "@/utils/localStorageUtils";
import { LOCAL_STORAGE_EVENT } from "@/utils/localStorageUtils";

function subscribe(callback: () => void) {
  const onStorage = (e: StorageEvent) => {
    if (e.key === "disableShowNewBadge") {
      callback();
    }
  };

  const onCustom = (e: Event) => {
    const { key } = (e as CustomEvent).detail;
    if (key === "disableShowNewBadge") {
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

function getSnapshot() {
  return getLocalStorageItem("disableShowNewBadge") === "true";
}

export function useDisableShowNewBadge() {
  return useSyncExternalStore(subscribe, getSnapshot);
}
