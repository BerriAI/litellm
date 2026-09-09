"use client";

import React, { useState } from "react";
import { ArrowUpCircle, X } from "lucide-react";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import {
  type LatestReleaseInfo,
  useLatestReleaseInfo,
} from "@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo";
import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";
import { isNewerVersion } from "@/utils/versionUtils";

const DISMISS_KEY_PREFIX = "litellm:upgradeBannerDismissed:";

interface UpgradeBannerProps {
  accessToken: string | null;
}

interface UpgradeBannerViewProps {
  currentVersion: string | null | undefined;
  latestRelease: LatestReleaseInfo | null | undefined;
}

const plural = (count: number, singular: string, pluralForm: string): string =>
  `${count} ${count === 1 ? singular : pluralForm}`;

export const describeRelease = ({ new_features, bug_fixes, other_updates }: LatestReleaseInfo): string =>
  [
    plural(new_features, "new feature", "new features"),
    plural(bug_fixes, "fix", "fixes"),
    `and ${plural(other_updates, "other update", "other updates")}`,
  ].join(", ");

export const UpgradeBannerView: React.FC<UpgradeBannerViewProps> = ({ currentVersion, latestRelease }) => {
  const [locallyDismissed, setLocallyDismissed] = useState(false);

  if (!currentVersion || !latestRelease || !isNewerVersion(currentVersion, latestRelease.version)) {
    return null;
  }

  const dismissKey = `${DISMISS_KEY_PREFIX}${latestRelease.version}`;
  if (locallyDismissed || getLocalStorageItem(dismissKey) === "true") {
    return null;
  }

  const handleClose = () => {
    setLocalStorageItem(dismissKey, "true");
    setLocallyDismissed(true);
  };

  return (
    <Alert variant="info" className="rounded-none border-x-0 border-t-0">
      <ArrowUpCircle className="size-4" aria-hidden />
      <AlertTitle>
        The latest version is{" "}
        <a href={latestRelease.release_url} target="_blank" rel="noopener noreferrer" className="underline">
          v{latestRelease.version}
        </a>
        : {describeRelease(latestRelease)}
      </AlertTitle>
      <AlertDescription>Your current version is v{currentVersion}</AlertDescription>
      <AlertAction>
        <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={handleClose}>
          <X className="size-4" />
        </Button>
      </AlertAction>
    </Alert>
  );
};

export const UpgradeBanner: React.FC<UpgradeBannerProps> = ({ accessToken }) => {
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const { data: latestRelease } = useLatestReleaseInfo(accessToken);
  return <UpgradeBannerView currentVersion={healthData?.litellm_version} latestRelease={latestRelease} />;
};
