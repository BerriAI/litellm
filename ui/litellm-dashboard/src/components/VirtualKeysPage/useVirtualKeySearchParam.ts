import { useCallback, useEffect, useState } from "react";

import { replaceWindowSearchParams } from "@/utils/replaceWindowSearchParams";

export const VIRTUAL_KEY_PARAM = "virtual_key";

function readVirtualKeyFromWindow(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return new URLSearchParams(window.location.search).get(VIRTUAL_KEY_PARAM);
}

/**
 * Deep-link the Virtual Keys page key detail view via ?virtual_key=<token hash>.
 * Local state drives the UI; history.replaceState keeps the address bar under /ui.
 */
export function useVirtualKeySearchParam() {
  const [virtualKeyId, setVirtualKeyIdState] = useState<string | null>(() => readVirtualKeyFromWindow());

  useEffect(() => {
    const syncFromWindow = () => {
      setVirtualKeyIdState(readVirtualKeyFromWindow());
    };
    window.addEventListener("popstate", syncFromWindow);
    return () => window.removeEventListener("popstate", syncFromWindow);
  }, []);

  const setVirtualKeyId = useCallback((token: string | null) => {
    setVirtualKeyIdState(token);
    replaceWindowSearchParams((params) => {
      if (!token) {
        params.delete(VIRTUAL_KEY_PARAM);
      } else {
        params.set(VIRTUAL_KEY_PARAM, token);
      }
    });
  }, []);

  return { virtualKeyId, setVirtualKeyId };
}
