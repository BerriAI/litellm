"use client";

import React from "react";
import Link from "next/link";
import { Alert } from "antd";

const DEPRECATION_DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/32090";
const DEPRECATION_TARGET_DATE = "September 1, 2026";

interface DeprecationBannerProps {
  featureName: string;
}

export const DeprecationBanner: React.FC<DeprecationBannerProps> = ({ featureName }) => (
  <Alert
    message={`${featureName} is on a draft deprecation list`}
    description={
      <>
        {`${featureName} is one of several experimental features we're considering removing, potentially as early as ${DEPRECATION_TARGET_DATE}. This list is a draft and is not final. If you rely on this feature, please share feedback on the `}
        <Link href={DEPRECATION_DISCUSSION_URL} target="_blank" rel="noopener noreferrer">
          deprecation discussion
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
