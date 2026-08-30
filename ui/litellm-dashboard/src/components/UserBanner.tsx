"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CircleAlert, Info, TriangleAlert, X } from "lucide-react";
import { Alert, AlertAction, AlertDescription } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { useUserBanner } from "@/app/(dashboard)/hooks/userBanner/useUserBanner";
import { UserBanner as UserBannerData, UserBannerSeverity } from "@/components/networking";

const DISMISS_STORAGE_KEY = "litellm:userBannerDismissed";

export const SEVERITY_ICONS: Record<UserBannerSeverity, React.ReactElement> = {
  info: <Info />,
  warning: <TriangleAlert />,
  error: <CircleAlert />,
};

export const bannerSignature = (banner: UserBannerData): string =>
  JSON.stringify({ message: banner.message, severity: banner.severity, revision: banner.revision });

export const UserBannerMarkdown: React.FC<{ message: string }> = ({ message }) => (
  <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    components={{
      a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
    }}
  >
    {message}
  </ReactMarkdown>
);

interface UserBannerProps {
  accessToken: string | null;
}

export const UserBanner: React.FC<UserBannerProps> = ({ accessToken }) => {
  const { data: banner } = useUserBanner(accessToken);
  const [dismissedSignature, setDismissedSignature] = useState<string | null>(() =>
    typeof window === "undefined" ? null : localStorage.getItem(DISMISS_STORAGE_KEY),
  );

  if (!banner?.enabled || banner.message.trim() === "") {
    return null;
  }

  const signature = bannerSignature(banner);
  if (dismissedSignature === signature) {
    return null;
  }

  const handleDismiss = () => {
    localStorage.setItem(DISMISS_STORAGE_KEY, signature);
    setDismissedSignature(signature);
  };

  return (
    <Alert variant={banner.severity} className="rounded-none border-x-0 border-t-0">
      {SEVERITY_ICONS[banner.severity]}
      <AlertDescription>
        <UserBannerMarkdown message={banner.message} />
      </AlertDescription>
      <AlertAction>
        <Button variant="ghost" size="icon-sm" aria-label="Dismiss banner" onClick={handleDismiss}>
          <X />
        </Button>
      </AlertAction>
    </Alert>
  );
};
