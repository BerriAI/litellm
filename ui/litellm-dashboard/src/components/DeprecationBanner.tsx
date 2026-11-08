"use client";

import React from "react";
import { Trans, useTranslation } from "react-i18next";
import Link from "next/link";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const DEPRECATION_DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/32090";

interface DeprecationBannerProps {
  featureName: string;
}

export const DeprecationBanner: React.FC<DeprecationBannerProps> = ({ featureName }) => {
  const { t } = useTranslation();
  return (
    <Alert className="mb-4">
      <Info />
      <AlertTitle>{t("shell.deprecationBanner.title", { featureName })}</AlertTitle>
      <AlertDescription>
        <Trans
          i18nKey="shell.deprecationBanner.description"
          components={{
            feature: <>{featureName}</>,
            discussion: <Link href={DEPRECATION_DISCUSSION_URL} target="_blank" rel="noopener noreferrer" />,
          }}
        />
      </AlertDescription>
    </Alert>
  );
};
