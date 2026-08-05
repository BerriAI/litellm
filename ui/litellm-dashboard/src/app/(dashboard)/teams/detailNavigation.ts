import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { navigateWithParams } from "../navigateWithParams";

export interface TeamDetailRouting {
  teamId: string | null;
  openTeam: (id: string) => void;
  close: () => void;
}

export function useTeamDetailRouting(): TeamDetailRouting {
  const searchParams = useSearchParams();

  const openTeam = useCallback((id: string) => {
    navigateWithParams((params) => {
      params.set("team", id);
    });
  }, []);

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete("team");
    });
  }, []);

  return {
    teamId: searchParams?.get("team") ?? null,
    openTeam,
    close,
  };
}
