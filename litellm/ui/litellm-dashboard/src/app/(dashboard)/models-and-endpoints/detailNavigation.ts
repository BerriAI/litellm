import { parseAsString, useQueryState, useQueryStates } from "nuqs";
import { useCallback } from "react";
import { useResetDetailState } from "@/hooks/useUrlTab";

export interface ModelDetailRouting {
  modelId: string | null;
  teamId: string | null;
  openModel: (id: string) => void;
  openTeam: (id: string) => void;
  close: () => void;
}

export function useModelDetailRouting(): ModelDetailRouting {
  const resetDetailState = useResetDetailState();
  const [{ model, team }, setParams] = useQueryStates(
    { model: parseAsString, team: parseAsString },
    { history: "push" },
  );

  const openModel = useCallback(
    (id: string) => {
      resetDetailState();
      void setParams({ model: id, team: null });
    },
    [setParams, resetDetailState],
  );

  const openTeam = useCallback(
    (id: string) => {
      resetDetailState();
      void setParams({ model: null, team: id });
    },
    [setParams, resetDetailState],
  );

  const close = useCallback(() => {
    resetDetailState();
    void setParams({ model: null, team: null });
  }, [setParams, resetDetailState]);

  return {
    modelId: model,
    teamId: team,
    openModel,
    openTeam,
    close,
  };
}

export interface ModelGroupFilterRouting {
  modelGroup: string | null;
  setModelGroup: (modelGroup: string | null) => void;
}

export function useModelGroupFilterRouting(): ModelGroupFilterRouting {
  const [modelGroup, setParam] = useQueryState("model_group", parseAsString);

  const setModelGroup = useCallback(
    (next: string | null) => {
      void setParam(next);
    },
    [setParam],
  );

  return { modelGroup, setModelGroup };
}
