import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

export interface ModelDetailRouting {
  modelId: string | null;
  teamId: string | null;
  openModel: (id: string) => void;
  openTeam: (id: string) => void;
  close: () => void;
}

/**
 * The dashboard is a static export served under /ui by the proxy, a prefix the
 * Next router (basePath "") does not know about. A router.push to the current
 * pathname with only the query changed is deduped and never re-renders, so the
 * detail overlay is driven by real browser navigation instead.
 */
function navigateWithParams(mutate: (params: URLSearchParams) => void): void {
  const params = new URLSearchParams(window.location.search);
  mutate(params);
  const qs = params.toString();
  window.location.assign(qs ? `${window.location.pathname}?${qs}` : window.location.pathname);
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
