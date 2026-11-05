import type { QueryClient } from "@tanstack/react-query";
import { modelDeprecationKeys } from "./useModelDeprecations";

export const invalidateModelQueries = async (queryClient: QueryClient): Promise<void> => {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: ["models", "list"] }),
    queryClient.invalidateQueries({ queryKey: modelDeprecationKeys.all }),
  ]);
};
