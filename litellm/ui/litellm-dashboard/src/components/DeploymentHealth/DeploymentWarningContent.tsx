import type { ParseKeys } from "i18next";
import type { ReactNode } from "react";
import { Trans } from "react-i18next";
import i18n from "@/i18n/i18n";
import { formatExpiryDate, getDaysUntilExpiration } from "@/utils/licenseUtils";

const LICENSE_EXPIRY_PREFIX = "license_expiry:";
const SALES_EMAIL = "sales@berri.ai";

export interface DeploymentWarningContent {
  title: string;
  fix?: ReactNode;
  docsUrl?: string;
}

const STATIC_CONTENT: Readonly<Record<string, DeploymentWarningContent & { title: ParseKeys }>> = {
  env_credential_login: {
    title: "shell.deploymentWarnings.envCredentialLogin.title",
    fix: (
      <Trans
        i18nKey="shell.deploymentWarnings.envCredentialLogin.fix"
        components={{
          username: <code>UI_USERNAME</code>,
          password: <code>UI_PASSWORD</code>,
          setting: <code>general_settings.disable_env_credential_login: true</code>,
        }}
      />
    ),
    docsUrl: "https://docs.litellm.ai/docs/proxy/ui",
  },
  no_redis: {
    title: "shell.deploymentWarnings.noRedis.title",
    fix: <Trans i18nKey="shell.deploymentWarnings.noRedis.fix" />,
    docsUrl: "https://docs.litellm.ai/docs/proxy/redis_requirements",
  },
  detailed_debug: {
    title: "shell.deploymentWarnings.detailedDebug.title",
    fix: (
      <Trans
        i18nKey="shell.deploymentWarnings.detailedDebug.fix"
        components={{ loglevel: <code>LITELLM_LOG=DEBUG</code> }}
      />
    ),
    docsUrl: "https://docs.litellm.ai/docs/proxy/prod",
  },
};

const describeCountdown = (days: number): string => {
  if (days <= 0) return i18n.t("shell.deploymentWarnings.license.expiresToday");
  if (days === 1) return i18n.t("shell.deploymentWarnings.license.expiresIn", { count: 1 });
  return i18n.t("shell.deploymentWarnings.license.expiresIn", { count: days });
};

const licenseExpiryContent = (expirationDate: string, now: Date): DeploymentWarningContent => {
  const days = getDaysUntilExpiration(expirationDate, now) ?? 0;
  const formattedDate = formatExpiryDate(expirationDate);
  const salesLink = <a href={`mailto:${SALES_EMAIL}`}>{SALES_EMAIL}</a>;
  return days < 0
    ? {
        title: i18n.t("shell.deploymentWarnings.license.expiredTitle", { date: formattedDate }),
        fix: <Trans i18nKey="shell.deploymentWarnings.license.expiredFix" components={{ sales: salesLink }} />,
        docsUrl: "https://docs.litellm.ai/docs/enterprise",
      }
    : {
        title: i18n.t("shell.deploymentWarnings.license.expiringTitle", {
          countdown: describeCountdown(days),
          date: formattedDate,
        }),
        fix: <Trans i18nKey="shell.deploymentWarnings.license.expiringFix" components={{ sales: salesLink }} />,
        docsUrl: "https://docs.litellm.ai/docs/enterprise",
      };
};

export const deploymentWarningContent = (warningId: string, now: Date = new Date()): DeploymentWarningContent => {
  if (warningId.startsWith(LICENSE_EXPIRY_PREFIX)) {
    return licenseExpiryContent(warningId.slice(LICENSE_EXPIRY_PREFIX.length), now);
  }
  if (!Object.hasOwn(STATIC_CONTENT, warningId)) {
    return { title: warningId };
  }
  const content = STATIC_CONTENT[warningId];
  return { ...content, title: i18n.t(content.title) };
};
