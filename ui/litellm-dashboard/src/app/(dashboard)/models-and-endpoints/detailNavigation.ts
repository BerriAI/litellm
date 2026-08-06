import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

import { useDetailHistoryClose } from "@/app/(dashboard)/hooks/useDetailHistoryClose";

export interface ModelDetailRouting {
  modelId: string | null;
  teamId: string | null;
  openModel: (id: string) => void;
  openTeam: (id: string) => void;
  close: () => void;
}

export function useModelDetailRouting(): ModelDetailRouting {
  const [{ model, team }, setParams] = useQueryStates(
    { model: parseAsString, team: parseAsString },
    { history: "push" },
  );

  const clearParams = useCallback(() => {
    void setParams({ model: null, team: null }, { history: "replace" });
  }, [setParams]);
  const { markOpened, close } = useDetailHistoryClose(clearParams);

  const openModel = useCallback(
    (id: string) => {
      markOpened();
      void setParams({ model: id, team: null });
    },
    [markOpened, setParams],
  );

  const openTeam = useCallback(
    (id: string) => {
      markOpened();
      void setParams({ model: null, team: id });
    },
    [markOpened, setParams],
  );

  return {
    modelId: model,
    teamId: team,
    openModel,
    openTeam,
    close,
  };
}
