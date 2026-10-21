import { useSyncExternalStore } from "react";
import {
  emitLocalStorageChange,
  getLocalStorageItem,
  LOCAL_STORAGE_EVENT,
  setLocalStorageItem,
} from "@/utils/localStorageUtils";

function subscribe(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener(LOCAL_STORAGE_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(LOCAL_STORAGE_EVENT, callback);
  };
}

export function useAnnouncementRead(readKey: string): boolean {
  return useSyncExternalStore(
    subscribe,
    () => getLocalStorageItem(readKey) === "true",
    () => false,
  );
}

export function markAnnouncementRead(readKey: string): void {
  setLocalStorageItem(readKey, "true");
  emitLocalStorageChange(readKey);
}

export function useUnreadAnnouncementCount(readKeys: readonly string[]): number {
  return useSyncExternalStore(
    subscribe,
    () => readKeys.filter((readKey) => getLocalStorageItem(readKey) !== "true").length,
    () => readKeys.length,
  );
}
