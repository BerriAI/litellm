"use client";

import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useLensApi } from "../../data/LensServices";
import { lensKeys, lensQueries } from "../../data/queries";
import { activeJob } from "../../model/status";
import type { Lens } from "../../model/types";

/** Run history polls off its own rows, so a job the list poll discovers has to push it to refetch. */
export function useRunHistory(lens: Lens, historyOffset: number, poll = true) {
  const api = useLensApi();
  const client = useQueryClient();
  const activeId = activeJob(lens.jobs)?.id;
  useEffect(() => {
    if (!poll || !activeId) return;
    void client.invalidateQueries({ queryKey: lensKeys.histories() }, { cancelRefetch: false });
  }, [activeId, client, poll]);
  const options = lensQueries.history(api, { lensId: lens.id, historyOffset });
  return useQuery({ ...options, refetchInterval: poll ? options.refetchInterval : false });
}
