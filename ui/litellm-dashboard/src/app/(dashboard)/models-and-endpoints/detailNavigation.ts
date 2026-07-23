import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

export interface ModelDetailRouting {
  modelId: string | null;
  teamId: string | null;
  openModel: (id: string) => void;
  openTeam: (id: string) => void;
  close: () => void;
}

function navigateWithParams(mutate: (params: URLSearchParams) => void): void {
  const params = new URLSearchParams(window.location.search);
  mutate(params);
  const qs = params.toString();
  const url = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
  window.history.pushState(null, "", url);
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
