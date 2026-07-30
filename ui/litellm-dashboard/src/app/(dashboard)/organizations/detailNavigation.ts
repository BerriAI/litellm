import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { navigateWithParams } from "../navigateWithParams";

export interface OrgDetailRouting {
  orgId: string | null;
  openOrg: (id: string) => void;
  close: () => void;
}

export function useOrgDetailRouting(): OrgDetailRouting {
  const searchParams = useSearchParams();

  const openOrg = useCallback((id: string) => {
    navigateWithParams((params) => {
      params.set("org", id);
    });
  }, []);

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete("org");
    });
  }, []);

  return {
    orgId: searchParams?.get("org") ?? null,
    openOrg,
    close,
  };
}
