import { LOCAL_STORAGE_EVENT, getLocalStorageItem } from "@/utils/localStorageUtils";
import { useSyncExternalStore } from "react";

function subscribe(callback: () => void) {
  const onStorage = (e: StorageEvent) => {
    if (e.key === "disableBlogPosts") {
      callback();
    }
  };

  const onCustom = (e: Event) => {
    const { key } = (e as CustomEvent).detail;
    if (key === "disableBlogPosts") {
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
  return getLocalStorageItem("disableBlogPosts") === "true";
}

export function useDisableBlogPosts() {
  return useSyncExternalStore(subscribe, getSnapshot);
}
