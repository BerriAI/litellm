import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { navigateWithParams } from "../navigateWithParams";

export interface ModelDetailRouting {
  modelId: string | null;
  teamId: string | null;
  openModel: (id: string) => void;
  openTeam: (id: string) => void;
  close: () => void;
}

export function useModelDetailRouting(): ModelDetailRouting {
  const searchParams = useSearchParams();

  const openModel = useCallback((id: string) => {
    navigateWithParams((params) => {
      params.delete("team");
      params.set("model", id);
    });
  }, []);

  const openTeam = useCallback((id: string) => {
    navigateWithParams((params) => {
      params.delete("model");
      params.set("team", id);
    });
  }, []);

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete("model");
      params.delete("team");
    });
  }, []);

  return {
    modelId: searchParams?.get("model") ?? null,
    teamId: searchParams?.get("team") ?? null,
    openModel,
    openTeam,
    close,
  };
}
