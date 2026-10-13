"use client";

import React from "react";
import Link from "next/link";
import { Info } from "lucide-react";

const DEPRECATION_DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/32090";

interface DeprecationBannerProps {
  featureName: string;
}

export const DeprecationBanner: React.FC<DeprecationBannerProps> = ({ featureName }) => (
  <div
    role="alert"
    className="mb-4 flex items-start gap-3 rounded-lg border border-border bg-muted/50 px-4 py-3 text-sm"
  >
    <Info className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
    <div className="min-w-0 flex-1">
      <p className="font-medium">{`${featureName} is deprecated`}</p>
      <p className="mt-1 break-words text-muted-foreground">
        {`${featureName} will be removed in the near future. See the `}
        <Link
          href={DEPRECATION_DISCUSSION_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-4"
        >
          deprecation discussion
        </Link>
        {" for details."}
      </p>
    </div>
  </div>
);
