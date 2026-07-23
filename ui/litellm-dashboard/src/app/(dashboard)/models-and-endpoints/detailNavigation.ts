import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

export interface ModelDetailRouting {
  modelId: string | null;
  teamId: string | null;
  openModel: (id: string) => void;
  openTeam: (id: string) => void;
  close: () => void;
}

export function useModelDetailRouting(): ModelDetailRouting {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const openModel = useCallback(
    (id: string) => {
      const params = new URLSearchParams(searchParams?.toString());
      params.delete("team");
      params.set("model", id);
      router.push(`${pathname}?${params.toString()}`);
    },
    [router, pathname, searchParams],
  );

  const openTeam = useCallback(
    (id: string) => {
      const params = new URLSearchParams(searchParams?.toString());
      params.delete("model");
      params.set("team", id);
      router.push(`${pathname}?${params.toString()}`);
    },
    [router, pathname, searchParams],
  );

  const close = useCallback(() => {
    const params = new URLSearchParams(searchParams?.toString());
    params.delete("model");
    params.delete("team");
    const qs = params.toString();
    router.push(qs ? `${pathname}?${qs}` : pathname);
  }, [router, pathname, searchParams]);

  return {
    modelId: searchParams?.get("model") ?? null,
    teamId: searchParams?.get("team") ?? null,
    openModel,
    openTeam,
    close,
  };
}
