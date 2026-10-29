"use client";

import React from "react";
import Link from "next/link";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const DEPRECATION_DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/32090";

interface DeprecationBannerProps {
  featureName: string;
}

export const DeprecationBanner: React.FC<DeprecationBannerProps> = ({ featureName }) => (
  <Alert className="mb-4">
    <Info />
    <AlertTitle>{`${featureName} is deprecated`}</AlertTitle>
    <AlertDescription>
      {`${featureName} will be removed in the near future. See the `}
      <Link href={DEPRECATION_DISCUSSION_URL} target="_blank" rel="noopener noreferrer">
        deprecation discussion
      </Link>
      {" for details."}
    </AlertDescription>
  </Alert>
);
