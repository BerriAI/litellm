// hooks/useHideAgentPlatformBanner.ts
import { useSyncExternalStore } from "react";
import { getLocalStorageItem, LOCAL_STORAGE_EVENT } from "@/utils/localStorageUtils";

export const HIDE_AGENT_PLATFORM_BANNER_KEY = "litellmHideAgentPlatformBanner";

function subscribe(callback: () => void) {
  const onStorage = (e: StorageEvent) => {
    if (e.key === HIDE_AGENT_PLATFORM_BANNER_KEY) {
      callback();
    }
  };

  const onCustom = (e: Event) => {
    const { key } = (e as CustomEvent).detail;
    if (key === HIDE_AGENT_PLATFORM_BANNER_KEY) {
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
  return getLocalStorageItem(HIDE_AGENT_PLATFORM_BANNER_KEY) === "true";
}

export function useHideAgentPlatformBanner() {
  return useSyncExternalStore(subscribe, getSnapshot);
}
