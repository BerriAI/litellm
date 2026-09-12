"use client";

import React from "react";
import { TriangleAlert } from "lucide-react";
import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";

const REDIS_DOCS_URL = "https://docs.litellm.ai/docs/proxy/redis_requirements";

interface NoRedisWarningBannerProps {
  accessToken: string | null;
}

export const NoRedisWarningBanner: React.FC<NoRedisWarningBannerProps> = ({ accessToken }) => {
  const { data: healthData } = useHealthReadinessDetails(accessToken);

  if (!healthData?.show_no_redis_warning) {
    return null;
  }

  return (
    <div
      role="alert"
      className="flex items-start gap-3 border-b border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
    >
      <TriangleAlert className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
      <div>
        <p className="font-semibold">No Redis configured. Redis is highly recommended</p>
        <p>
          This proxy is running more than one worker (or the worker count could not be verified). Without Redis, rate
          limits, budgets, router state, and cache invalidation are per worker, so limits are enforced once per worker
          and spend can overshoot.{" "}
          <a className="underline" href={REDIS_DOCS_URL} target="_blank" rel="noreferrer">
            See everything that does not work without Redis
          </a>
          . Set <code className="font-mono">LITELLM_DISABLE_NO_REDIS_WARNING=true</code> to hide this banner anyway.
        </p>
      </div>
    </div>
  );
};
