import { parseAsString, useQueryStates } from "nuqs";
import { useCallback } from "react";

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

  const openModel = useCallback(
    (id: string) => {
      void setParams({ model: id, team: null });
    },
    [setParams],
  );

  const openTeam = useCallback(
    (id: string) => {
      void setParams({ model: null, team: id });
    },
    [setParams],
  );

  const close = useCallback(() => {
    void setParams({ model: null, team: null });
  }, [setParams]);

  return {
    modelId: model,
    teamId: team,
    openModel,
    openTeam,
    close,
  };
}
