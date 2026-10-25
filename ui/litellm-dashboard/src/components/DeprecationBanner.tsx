"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import Link from "next/link";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const DEPRECATION_DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/32090";

interface DeprecationBannerProps {
  featureName: string;
}

export const DeprecationBanner: React.FC<DeprecationBannerProps> = ({ featureName }) => {
  const { t } = useLanguage();
  return (
    <Alert className="mb-4">
      <Info />
      <AlertTitle>{t("{featureName} is deprecated", { featureName: String(featureName) })}</AlertTitle>
      <AlertDescription>
        {t("{featureName} will be removed in the near future. See the ", { featureName: String(featureName) })}
        <Link href={DEPRECATION_DISCUSSION_URL} target="_blank" rel="noopener noreferrer">
          {t("deprecation discussion")}
        </Link>
        {t(" for details.")}
      </AlertDescription>
    </Alert>
  );
};
