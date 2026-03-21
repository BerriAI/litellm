// hooks/useDisableShowPrompts.ts
import { useSyncExternalStore } from "react";
import { getLocalStorageItem } from "@/utils/localStorageUtils";
import { LOCAL_STORAGE_EVENT } from "@/utils/localStorageUtils";

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

function getSnapshot() {
  return getLocalStorageItem("disableShowPrompts") === "true";
}

export function useDisableShowPrompts() {
  return useSyncExternalStore(subscribe, getSnapshot);
}
