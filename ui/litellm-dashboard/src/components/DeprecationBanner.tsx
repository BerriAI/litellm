"use client";

import React from "react";
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
        {featureName} is one of several experimental features we&apos;re considering removing, potentially as early as
        September 1, 2026. This list is a draft and is not final. If you rely on this feature, please share feedback on
        the{" "}
        <a href={DEPRECATION_DISCUSSION_URL} target="_blank" rel="noopener noreferrer">
          deprecation discussion
        </a>
        .
      </>
    }
    type="info"
    showIcon
    closable
    style={{ marginBottom: 16 }}
  />
);

export default DeprecationBanner;
