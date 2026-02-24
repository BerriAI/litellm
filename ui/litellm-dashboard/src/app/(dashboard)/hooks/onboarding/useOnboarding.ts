import { claimOnboardingToken, getOnboardingCredentials } from "@/components/networking";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useUIConfig } from "../uiConfig/useUIConfig";
import { createQueryKeys } from "../common/queryKeysFactory";

const onboardingKeys = createQueryKeys("onboarding");

export const useOnboardingCredentials = (inviteId: string | null) => {
  const { isLoading: isUIConfigLoading } = useUIConfig();
  return useQuery({
    queryKey: onboardingKeys.detail(inviteId ?? ""),
    queryFn: async () => await getOnboardingCredentials(inviteId!),
    enabled: Boolean(inviteId) && !isUIConfigLoading,
  });
};

export interface ClaimTokenParams {
  accessToken: string;
  inviteId: string;
  userId: string;
  password: string;
}

export const useClaimOnboardingToken = () => {
  return useMutation({
    mutationFn: async ({ accessToken, inviteId, userId, password }: ClaimTokenParams) =>
      await claimOnboardingToken(accessToken, inviteId, userId, password),
  });
};
