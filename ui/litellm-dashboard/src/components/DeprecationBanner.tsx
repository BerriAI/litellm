"use client";

import React from "react";
import Link from "next/link";
import { Alert } from "antd";
import { useTranslation } from "react-i18next";

const DEPRECATION_DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/32090";

const FEATURE_TRANSLATION_KEYS: Record<string, string> = {
  "The Playground's Agent Builder": "agentBuilder",
  Workflows: "workflows",
  "The API Reference tab": "apiReference",
  "MCP Network Settings and the internal-network-only flag": "mcpNetwork",
  Memory: "memory",
  "The old Usage page": "oldUsage",
  "Prompt Management": "prompts",
};

interface DeprecationBannerProps {
  featureName: string;
}

export const DeprecationBanner: React.FC<DeprecationBannerProps> = ({ featureName }) => {
  const { t } = useTranslation("common");
  const featureKey = FEATURE_TRANSLATION_KEYS[featureName];
  const localizedFeature = featureKey ? t(`deprecation.features.${featureKey}`) : featureName;

  return (
    <Alert
      message={t("deprecation.title", { feature: localizedFeature })}
      description={
        <>
          {t("deprecation.description", {
            feature: localizedFeature,
            date: t("deprecation.targetDate"),
          })}{" "}
          <Link href={DEPRECATION_DISCUSSION_URL} target="_blank" rel="noopener noreferrer">
            {t("deprecation.discussion")}
          </Link>
          .
        </>
      }
      type="info"
      showIcon
      closable
      style={{ marginBottom: 16 }}
    />
  );
};
