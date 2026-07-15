import { useMutation, useQuery, useQueryClient, UseQueryResult } from "@tanstack/react-query";
import { ptuReservationCloseCall, ptuReservationCreateCall, ptuReservationListCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";

export interface PtuReservationItem {
  id: string;
  team_id: string;
  model: string;
  cost_source: "manual" | "azure_billing";
  ptu_count: number | null;
  cost_per_ptu: number | null;
  azure_resource_id: string | null;
  effective_from: string;
  effective_to: string | null;
  created_by: string;
  created_at: string;
  updated_by: string;
  updated_at: string;
}

export interface PtuReservationListFilters {
  team_id?: string;
  model?: string;
  active_only?: boolean;
}

export const ptuReservationKeys = createQueryKeys("ptuReservations");

const filtersToListParam = (filters: PtuReservationListFilters): Record<string, string | number> => {
  const out: Record<string, string | number> = {};
  if (filters.team_id) out.team_id = filters.team_id;
  if (filters.model) out.model = filters.model;
  if (filters.active_only) out.active_only = "true";
  return out;
};

export const usePtuReservations = (
  filters: PtuReservationListFilters = {},
  options: { enabled?: boolean } = {},
): UseQueryResult<PtuReservationItem[]> => {
  const { accessToken } = useAuthorized();
  return useQuery<PtuReservationItem[]>({
    queryKey: ptuReservationKeys.list({ filters: filtersToListParam(filters) }),
    queryFn: async () => {
      const data = await ptuReservationListCall(accessToken!, filters);
      return (data ?? []).filter((item: PtuReservationItem | null): item is PtuReservationItem => item != null);
    },
    enabled: Boolean(accessToken) && options.enabled !== false,
  });
};

export const useCreatePtuReservation = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<unknown, Error, Record<string, any>>({
    mutationFn: async (formValues) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return ptuReservationCreateCall(accessToken, formValues);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ptuReservationKeys.all });
    },
  });
};

export const useClosePtuReservation = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<unknown, Error, { id: string; effective_to?: string }>({
    mutationFn: async ({ id, effective_to }) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return ptuReservationCloseCall(accessToken, id, effective_to);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ptuReservationKeys.all });
    },
  });
};
