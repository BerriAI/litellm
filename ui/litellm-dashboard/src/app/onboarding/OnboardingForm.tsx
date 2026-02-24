"use client";
import React from "react";
import { useSearchParams } from "next/navigation";
import { jwtDecode } from "jwt-decode";
import { useOnboardingCredentials, useClaimOnboardingToken } from "@/app/(dashboard)/hooks/onboarding/useOnboarding";
import { getProxyBaseUrl } from "@/components/networking";
import { OnboardingLoadingView } from "./OnboardingLoadingView";
import { OnboardingErrorView } from "./OnboardingErrorView";
import { OnboardingFormBody } from "./OnboardingFormBody";

type OnboardingFormProps = {
  variant: "signup" | "reset_password";
};

export function OnboardingForm({ variant }: OnboardingFormProps) {
  const searchParams = useSearchParams()!;
  const inviteId = searchParams.get("invitation_id");
  const [claimError, setClaimError] = React.useState<string | null>(null);

  const {
    data: credentialsData,
    isLoading: isCredentialsLoading,
    isError: isCredentialsError,
  } = useOnboardingCredentials(inviteId);

  const { mutate: claimToken, isPending } = useClaimOnboardingToken();

  const decoded = credentialsData?.token
    ? (jwtDecode(credentialsData.token) as { [key: string]: any })
    : null;
  const userEmail: string = decoded?.user_email ?? "";
  const userId: string | null = decoded?.user_id ?? null;
  const accessToken: string | null = decoded?.key ?? null;
  const jwtToken: string | null = credentialsData?.token ?? null;

  const handleSubmit = (formValues: { password: string }) => {
    if (!accessToken || !jwtToken || !userId || !inviteId) return;

    setClaimError(null);

    claimToken(
      { accessToken, inviteId, userId, password: formValues.password },
      {
        onSuccess: () => {
          document.cookie = `token=${jwtToken}; path=/; SameSite=Lax`;
          const proxyBaseUrl = getProxyBaseUrl();
          window.location.href = proxyBaseUrl
            ? `${proxyBaseUrl}/ui/?login=success`
            : "/ui/?login=success";
        },
        onError: (error: Error) => {
          setClaimError(error.message || "Failed to submit. Please try again.");
        },
      }
    );
  };

  if (isCredentialsLoading) return <OnboardingLoadingView />;
  if (isCredentialsError) return <OnboardingErrorView />;

  return (
    <OnboardingFormBody
      variant={variant}
      userEmail={userEmail}
      isPending={isPending}
      claimError={claimError}
      onSubmit={handleSubmit}
    />
  );
}
