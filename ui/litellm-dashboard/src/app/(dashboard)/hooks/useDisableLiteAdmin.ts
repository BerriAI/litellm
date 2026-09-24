import { useSyncExternalStore } from "react";
import { getProxyBaseUrl } from "@/components/networking";
import {
  LOCAL_STORAGE_EVENT,
  emitLocalStorageChange,
  getLocalStorageItem,
  removeLocalStorageItem,
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

export function useDisableLiteAdmin(userId: string | null) {
  const key = userId ? `disableLiteAdmin:${JSON.stringify([getProxyBaseUrl(), userId])}` : null;
  const disabled = useSyncExternalStore(
    subscribe,
    () => key !== null && getLocalStorageItem(key) === "true",
    () => false,
  );
  const setDisabled = (value: boolean) => {
    if (key === null) return;
    if (value) setLocalStorageItem(key, "true");
    else removeLocalStorageItem(key);
    emitLocalStorageChange(key);
  };
  return [disabled, setDisabled] as const;
}
