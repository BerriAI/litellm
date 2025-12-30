import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { getGuardrailsList } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const guardrailKeys = createQueryKeys("guardrails");

export const useGuardrails = (): UseQueryResult<string[]> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<string[]>({
    queryKey: guardrailKeys.list({}),
    queryFn: async () => {
      const response = await getGuardrailsList(accessToken!);
      return response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
    },
    enabled: Boolean(accessToken && userId && userRole),
  });
};
