import { ComplexityScorerDefaults, getComplexityScorerDefaults } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const scorerDefaultsKeys = createQueryKeys("complexityScorerDefaults");

export const useComplexityScorerDefaults = () => {
  // 24 hours: the shipped defaults only change on a release.
  const options = {
    queryKey: scorerDefaultsKeys.list({}),
    queryFn: async () => await getComplexityScorerDefaults(),
    staleTime: 24 * 60 * 60 * 1000,
    gcTime: 24 * 60 * 60 * 1000,
  };
  return useQuery<ComplexityScorerDefaults>(options);
};
