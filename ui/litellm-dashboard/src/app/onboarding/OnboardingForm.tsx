"use client";
import React from "react";
import { useSearchParams } from "next/navigation";
import { jwtDecode } from "jwt-decode";
import { useOnboardingCredentials, useClaimOnboardingToken } from "@/app/(dashboard)/hooks/onboarding/useOnboarding";
import { getProxyBaseUrl } from "@/components/networking";
import { clearTokenCookies, storeLoginToken } from "@/utils/cookieUtils";
import { OnboardingLoadingView } from "./OnboardingLoadingView";
import { OnboardingErrorView } from "./OnboardingErrorView";
import { OnboardingFormBody } from "./OnboardingFormBody";
import LanguageSelector from "@/components/LanguageSelector/LanguageSelector";
import { useTranslation } from "react-i18next";

type OnboardingFormProps = {
  variant: "signup" | "reset_password";
};

export function OnboardingForm({ variant }: OnboardingFormProps) {
  const { t } = useTranslation("auth");
  const searchParams = useSearchParams()!;
  const inviteId = searchParams.get("invitation_id");
  const [claimError, setClaimError] = React.useState<string | null>(null);

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
            setClaimError(t("onboarding.sessionStartError"));
            return;
          }
          // Invite signup is a principal-change boundary — the prior admin's
          // cookies/sessionStorage must be invalidated before the new user's
          // session is established, otherwise getCookie() can fall back to
          // the inviter's token.
          clearTokenCookies();
          storeLoginToken(data.token);
          const proxyBaseUrl = getProxyBaseUrl();
          window.location.href = proxyBaseUrl ? `${proxyBaseUrl}/ui/?login=success` : "/ui/?login=success";
        },
        onError: (error: Error) => {
          setClaimError(error.message || t("onboarding.submitError"));
        },
      },
    );
  };

  const languageToolbar = (
    <div className="fixed right-4 top-4 z-50 rounded-lg border border-gray-200 bg-white p-1 shadow-sm">
      <LanguageSelector />
    </div>
  );

  if (isCredentialsLoading)
    return (
      <>
        {languageToolbar}
        <OnboardingLoadingView />
      </>
    );
  if (isCredentialsError)
    return (
      <>
        {languageToolbar}
        <OnboardingErrorView />
      </>
    );

  return (
    <>
      {languageToolbar}
      <OnboardingFormBody
        variant={variant}
        userEmail={userEmail}
        isPending={isPending}
        claimError={claimError}
        onSubmit={handleSubmit}
      />
    </>
  );
}
