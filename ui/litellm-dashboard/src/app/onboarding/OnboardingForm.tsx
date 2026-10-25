"use client";
import React from "react";
import { useSearchParams } from "next/navigation";
import { jwtDecode } from "jwt-decode";
import { useOnboardingCredentials, useClaimOnboardingToken } from "@/app/(dashboard)/hooks/onboarding/useOnboarding";
import { getProxyBaseUrl, getUiPageUrl } from "@/components/networking";
import { clearTokenCookies, storeLoginToken } from "@/utils/cookieUtils";
import { OnboardingLoadingView } from "./OnboardingLoadingView";
import { OnboardingErrorView } from "./OnboardingErrorView";
import { OnboardingFormBody } from "./OnboardingFormBody";
import TranslatedMessage from "@/components/shared/TranslatedMessage";

type OnboardingFormProps = {
  variant: "signup" | "reset_password";
};

export function OnboardingForm({ variant }: OnboardingFormProps) {
  const searchParams = useSearchParams()!;
  const inviteId = searchParams.get("invitation_id");
  // 本地提示延迟到渲染时翻译，服务端错误仍按原文显示。
  const [claimError, setClaimError] = React.useState<React.ReactNode>(null);

  const {
    data: credentialsData,
    isLoading: isCredentialsLoading,
    isError: isCredentialsError,
  } = useOnboardingCredentials(inviteId);

  const { mutate: claimToken, isPending } = useClaimOnboardingToken();

  const decoded = credentialsData?.token ? (jwtDecode(credentialsData.token) as { [key: string]: any }) : null;
  const userEmail: string = decoded?.user_email ?? "";
  const userId: string | null = decoded?.user_id ?? null;
  const accessToken: string | null = decoded?.key ?? null;

  const handleSubmit = (formValues: { password: string }) => {
    if (!accessToken || !userId || !inviteId) return;

    setClaimError(null);

    claimToken(
      { accessToken, inviteId, userId, password: formValues.password },
      {
        onSuccess: (data: { token?: string }) => {
          if (!data?.token) {
            setClaimError(<TranslatedMessage message="Failed to start session. Please try again." />);
            return;
          }
          // Invite signup is a principal-change boundary — the prior admin's
          // cookies/sessionStorage must be invalidated before the new user's
          // session is established, otherwise getCookie() can fall back to
          // the inviter's token.
          clearTokenCookies();
          storeLoginToken(data.token);
          const proxyBaseUrl = getProxyBaseUrl();
          window.location.href = getUiPageUrl(
            proxyBaseUrl ? `${proxyBaseUrl}/ui/?login=success` : "/ui/?login=success",
          );
        },
        onError: (error: Error) => {
          setClaimError(error.message || <TranslatedMessage message="Failed to submit. Please try again." />);
        },
      },
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
