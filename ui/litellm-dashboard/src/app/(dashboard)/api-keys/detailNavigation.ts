import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { navigateWithParams } from "../navigateWithParams";

export interface KeyDetailRouting {
  keyId: string | null;
  openKey: (id: string) => void;
  close: () => void;
}

export function useKeyDetailRouting(): KeyDetailRouting {
  const searchParams = useSearchParams();

  const openKey = useCallback((id: string) => {
    navigateWithParams((params) => {
      params.set("key", id);
    });
  }, []);

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete("key");
    });
  }, []);

  return {
    keyId: searchParams?.get("key") ?? null,
    openKey,
    close,
  };
}
