import { useMutation, useQueryClient } from "@tanstack/react-query";
import { keyKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";
import { toast } from "@/lib/toast";
import { keyUpdateCall } from "@/components/networking";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import type { BudgetQuickEditPayload, ModelsQuickEditPayload } from "./quickEditPayload";

type QuickKeyUpdateVariables =
  | { kind: "budget"; payload: BudgetQuickEditPayload }
  | { kind: "models"; payload: ModelsQuickEditPayload };

export const useQuickKeyUpdate = (accessToken: string) => {
  const queryClient = useQueryClient();

  return useMutation<Partial<KeyResponse>, unknown, QuickKeyUpdateVariables>({
    mutationFn: async ({ payload }) => (await keyUpdateCall(accessToken, payload)) as Partial<KeyResponse>,
    onSuccess: (_updated, variables) => {
      void queryClient.invalidateQueries({ queryKey: keyKeys.all });
      toast.success(variables.kind === "budget" ? "Budget updated" : "Models updated");
    },
    onError: (error) => toast.fromError(parseErrorMessage(error)),
  });
};
