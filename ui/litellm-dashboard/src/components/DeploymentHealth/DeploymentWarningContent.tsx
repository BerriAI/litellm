import type { ReactNode } from "react";
import { formatExpiryDate, getDaysUntilExpiration } from "@/utils/licenseUtils";

const LICENSE_EXPIRY_PREFIX = "license_expiry:";
const SALES_EMAIL = "sales@berri.ai";

export interface DeploymentWarningContent {
  title: string;
  fix?: ReactNode;
  docsUrl?: string;
}

const STATIC_CONTENT: Readonly<Record<string, DeploymentWarningContent>> = {
  env_credential_login: {
    title: "Environment-credential login is enabled",
    fix: (
      <>
        Anyone with <code>UI_USERNAME</code>/<code>UI_PASSWORD</code> (or the master key, when <code>UI_PASSWORD</code>{" "}
        is unset) can sign in as a proxy admin with a shared static secret. Create a regular admin account with its own
        password, then set <code>general_settings.disable_env_credential_login: true</code>
      </>
    ),
    docsUrl: "https://docs.litellm.ai/docs/proxy/ui",
  },
  no_redis: {
    title: "No Redis configured",
    fix: (
      <>
        This proxy runs more than one worker (or the worker count could not be verified). Without Redis, rate limits,
        budgets, router state and cache invalidation are per worker, so limits are enforced once per worker and spend
        can overshoot
      </>
    ),
    docsUrl: "https://docs.litellm.ai/docs/proxy/redis_requirements",
  },
  detailed_debug: {
    title: "Detailed debug logging is on",
    fix: (
      <>
        <code>LITELLM_LOG=DEBUG</code> logs extensive diagnostic information and significantly slows the proxy. Use it
        only while troubleshooting and turn it off in production
      </>
    ),
    docsUrl: "https://docs.litellm.ai/docs/proxy/prod",
  },
};

const describeCountdown = (days: number): string => {
  if (days <= 0) return "expires today";
  if (days === 1) return "expires in 1 day";
  return `expires in ${days} days`;
};

const licenseExpiryContent = (expirationDate: string, now: Date): DeploymentWarningContent => {
  const days = getDaysUntilExpiration(expirationDate, now) ?? 0;
  const formattedDate = formatExpiryDate(expirationDate);
  const salesLink = <a href={`mailto:${SALES_EMAIL}`}>{SALES_EMAIL}</a>;
  return days < 0
    ? {
        title: `Your LiteLLM Enterprise license expired on ${formattedDate}`,
        fix: <>Enterprise features are now disabled. Reach out to {salesLink} to restore access</>,
        docsUrl: "https://docs.litellm.ai/docs/proxy/enterprise",
      }
    : {
        title: `Your LiteLLM Enterprise license ${describeCountdown(days)} (${formattedDate})`,
        fix: <>Renew before it lapses to keep enterprise features. Reach out to {salesLink}</>,
        docsUrl: "https://docs.litellm.ai/docs/proxy/enterprise",
      };
};

export const deploymentWarningContent = (warningId: string, now: Date = new Date()): DeploymentWarningContent => {
  if (warningId.startsWith(LICENSE_EXPIRY_PREFIX)) {
    return licenseExpiryContent(warningId.slice(LICENSE_EXPIRY_PREFIX.length), now);
  }
  return STATIC_CONTENT[warningId] ?? { title: warningId };
};
