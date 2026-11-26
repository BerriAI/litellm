"use client";

import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import {
  type LatestReleaseInfo,
  useLatestReleaseInfo,
} from "@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo";
import {
  markAnnouncementRead,
  useAnnouncementRead,
  useUnreadAnnouncementCount,
} from "@/app/(dashboard)/hooks/useAnnouncementRead";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { DashboardResources } from "@/components/Navbar/DashboardResources";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/cva.config";
import { isNewerVersion } from "@/utils/versionUtils";
import { AlertTriangle, ArrowUpRight, Bell } from "lucide-react";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { metricValue, presentAlert, type AlertItem, type SavedAlert } from "@/components/Alerts/alertPresentation";
import { useAlerts } from "@/components/Alerts/useAlerts";
import { effectiveSessionRole, isProxyAdminTierRole } from "@/utils/roles";
import { uiHref } from "@/utils/uiHref";
import { type DeprecationAlert } from "@/components/ModelDeprecationAlerts/deprecationAlerts";
import { useDeprecationAlerts } from "@/components/ModelDeprecationAlerts/useDeprecationAlerts";
import { AlertStatus, effectiveAlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { migrationHref } from "@/components/ModelDeprecationAlerts/migrationRoute";
import React, { Fragment } from "react";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";

export const AUTO_ROUTER_DOCS_URL = "https://docs.litellm.ai/docs/proxy/auto_routing";

interface Announcement {
  readKey: string;
  title: string;
  description: string;
  linkLabel: string;
  href: string;
}

const autoRouterAnnouncement = (): Announcement => ({
  readKey: "litellmHideAutoRouterAnnouncement",
  title: i18n.t("shell.notifications.autoRouter.title"),
  description: i18n.t("shell.notifications.autoRouter.description"),
  linkLabel: i18n.t("shell.notifications.autoRouter.link"),
  href: AUTO_ROUTER_DOCS_URL,
});

const describeUpgrade = (
  currentVersion: string,
  { new_features, bug_fixes, other_updates }: LatestReleaseInfo,
): string => {
  if (new_features + bug_fixes + other_updates === 0) {
    return i18n.t("shell.notifications.upgrade.descriptionWithoutCounts", { current: currentVersion });
  }
  return i18n.t("shell.notifications.upgrade.description", {
    summary: [
      i18n.t("shell.notifications.release.newFeatures", { count: new_features }),
      i18n.t("shell.notifications.release.fixes", { count: bug_fixes }),
      i18n.t("shell.notifications.release.otherUpdates", { count: other_updates }),
    ].join(", "),
    current: currentVersion,
  });
};

const upgradeAnnouncement = (
  currentVersion: string | null | undefined,
  latestRelease: LatestReleaseInfo | null | undefined,
): Announcement | null => {
  if (!currentVersion || !latestRelease || !isNewerVersion(currentVersion, latestRelease.version)) {
    return null;
  }
  return {
    readKey: `litellm:upgradeBannerDismissed:${latestRelease.version}`,
    title: i18n.t("shell.notifications.upgrade.title", { version: latestRelease.version }),
    description: describeUpgrade(currentVersion, latestRelease),
    linkLabel: i18n.t("shell.notifications.upgrade.link"),
    href: latestRelease.release_url,
  };
};

const AnnouncementItem: React.FC<{ announcement: Announcement }> = ({ announcement }) => {
  const { t } = useTranslation();
  const read = useAnnouncementRead(announcement.readKey);
  return (
    <article aria-label={announcement.title} className="flex flex-col gap-2">
      <p className="text-sm font-medium">{announcement.title}</p>
      <p className="text-xs leading-relaxed text-muted-foreground">{announcement.description}</p>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <a
          className={cn(
            buttonVariants({ variant: "link", size: "sm" }),
            "h-auto gap-1 p-0 text-xs font-normal text-muted-foreground hover:text-foreground",
          )}
          href={announcement.href}
          target="_blank"
          rel="noopener noreferrer"
        >
          {announcement.linkLabel}
          <ArrowUpRight className="size-3" aria-hidden />
        </a>
        {read ? null : (
          <Button
            variant="link"
            size="sm"
            className="h-auto px-0! text-xs font-normal text-muted-foreground"
            onClick={() => markAnnouncementRead(announcement.readKey)}
          >
            {t("shell.notifications.announcement.markAsRead")}
          </Button>
        )}
      </div>
    </article>
  );
};

interface NotificationsBellViewProps {
  deprecationAlerts?: readonly DeprecationAlert[];
  performanceAlerts?: readonly SavedAlert[];
  performanceError?: boolean;
  deprecationError?: boolean;
  deploymentHealthLink?: React.ReactNode;
  currentVersion: string | null | undefined;
  latestRelease: LatestReleaseInfo | null | undefined;
}

export const NotificationsBellView: React.FC<NotificationsBellViewProps> = ({
  currentVersion,
  latestRelease,
  deploymentHealthLink,
  deprecationAlerts,
  performanceAlerts,
  performanceError,
  deprecationError,
}) => {
  const { t } = useTranslation();
  const upgrade = upgradeAnnouncement(currentVersion, latestRelease);
  const announcements = upgrade ? [upgrade, autoRouterAnnouncement()] : [autoRouterAnnouncement()];
  const unreadCount = useUnreadAnnouncementCount(announcements.map((announcement) => announcement.readKey));

  const items: AlertItem[] = [
    ...(deprecationAlerts ?? []).map((alert): AlertItem => ({ source: "deprecation", alert })),
    ...(performanceAlerts ?? []).map((alert): AlertItem => ({ source: "performance", alert })),
  ];
  const openAlerts = items
    .filter(
      (item) =>
        (item.source === "deprecation" ? effectiveAlertStatus(item.alert) : item.alert.status) === AlertStatus.Open,
    )
    .slice(0, 5)
    .map((item) => {
      const alert = presentAlert(item);
      const description =
        item.source === "performance" && "value" in item.alert.evidence
          ? t("shell.notifications.bell.comparison", {
              value: metricValue(item.alert.evidence.value, item.alert.evidence.unit),
              baseline: metricValue(item.alert.evidence.baseline, item.alert.evidence.unit),
            })
          : alert.description;
      const href =
        item.source === "deprecation"
          ? migrationHref(item.alert.usageModel)
          : `${uiHref("alerts")}?alert=${encodeURIComponent(alert.id)}`;
      return { id: `${item.source}:${alert.id}`, title: alert.title, description, href };
    });
  const hasAlerts = openAlerts.length > 0;
  const hasError = performanceError || deprecationError;
  const showAlerts = Boolean(deprecationAlerts || performanceAlerts || hasError);
  const announcementLabel =
    unreadCount > 0
      ? t("shell.notifications.bell.triggerUnread", { unread: unreadCount })
      : t("shell.notifications.bell.trigger");
  const label = hasAlerts ? t("shell.notifications.bell.triggerAlerts") : announcementLabel;

  return (
    <Popover>
      <PopoverTrigger
        render={<Button variant="ghost" size="icon" className="text-muted-foreground" />}
        aria-label={label}
      >
        <span className="relative inline-flex">
          <Bell className={cn("size-4", hasAlerts && "text-amber-700")} aria-hidden />
          {hasAlerts ? (
            <span className="absolute -top-2 -right-2 rounded-full bg-background p-px text-amber-600" aria-hidden>
              <AlertTriangle className="size-3" fill="currentColor" stroke="var(--background)" strokeWidth={1.5} />
            </span>
          ) : null}
          {!hasAlerts && unreadCount > 0 ? (
            <Badge className="absolute -top-0.5 -right-1 size-1.5 p-0" aria-hidden />
          ) : null}
        </span>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        aria-label={t("shell.notifications.bell.trigger")}
        className="w-96 max-w-[calc(100vw-1rem)] gap-0 overflow-hidden p-0"
      >
        <PopoverTitle className="border-b border-border px-4 py-3 text-sm font-semibold">
          {t("shell.notifications.bell.title")}
        </PopoverTitle>
        <div className="max-h-[min(32rem,calc(100dvh-10rem))] overflow-y-auto">
          <div className="space-y-4 p-4">
            {showAlerts && (
              <section aria-label={t("alerts.unified.title")}>
                <div className="flex items-center justify-between gap-3">
                  <h3 className="text-sm font-semibold">{t("alerts.unified.title")}</h3>
                  <a className="text-xs text-primary hover:underline" href={uiHref("alerts")}>
                    {t("shell.notifications.bell.seeAll")}
                  </a>
                </div>
                <ul className="mt-2 divide-y divide-border">
                  {openAlerts.map((alert) => (
                    <li key={alert.id}>
                      <a href={alert.href} className="-mx-2 block rounded-md px-2 py-3 hover:bg-accent">
                        <p className="line-clamp-2 break-words text-sm font-medium">{alert.title}</p>
                        <p className="mt-1 line-clamp-1 text-xs text-muted-foreground">{alert.description}</p>
                      </a>
                    </li>
                  ))}
                </ul>
                {deprecationError && (
                  <p className="mt-2 text-xs text-muted-foreground">{t("alerts.unified.deprecationsError")}</p>
                )}
                {performanceError && (
                  <p className="mt-2 text-xs text-muted-foreground">{t("alerts.unified.performanceError")}</p>
                )}
                {!hasAlerts && !hasError && (
                  <p className="mt-2 text-xs text-muted-foreground">{t("shell.notifications.bell.noOpenAlerts")}</p>
                )}
              </section>
            )}
            {showAlerts && (
              <>
                <Separator />
                <p className="text-xs font-medium text-muted-foreground">
                  {t("shell.notifications.bell.announcementsHeading")}
                </p>
              </>
            )}

            {announcements.map((announcement, index) => (
              <Fragment key={announcement.readKey}>
                {index > 0 ? <Separator /> : null}
                <AnnouncementItem announcement={announcement} />
              </Fragment>
            ))}
          </div>
          <DashboardResources>{deploymentHealthLink}</DashboardResources>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export const NotificationsBell: React.FC<{ accessToken: string | null; deploymentHealthLink?: React.ReactNode }> = ({
  accessToken,
  deploymentHealthLink,
}) => {
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const { data: latestRelease } = useLatestReleaseInfo(accessToken);
  const { data: currentUser, isError: userError } = useCurrentUser(accessToken);
  const authenticated = Boolean(accessToken && currentUser?.user_id && !userError);
  const allowed = authenticated && isProxyAdminTierRole(currentUser?.user_role ?? "");
  const scope = JSON.stringify([currentUser?.user_id, effectiveSessionRole(currentUser?.user_role ?? undefined)]);
  const performance = useAlerts(scope, allowed, AlertStatus.Open, "all");
  const deprecations = useDeprecationAlerts(accessToken);
  return (
    <NotificationsBellView
      currentVersion={healthData?.litellm_version}
      latestRelease={latestRelease}
      deprecationAlerts={deprecations.isError ? undefined : deprecations.data}
      performanceAlerts={allowed && !performance.isError ? performance.data?.pages[0]?.data : undefined}
      performanceError={allowed && performance.isError}
      deprecationError={deprecations.isError}
      deploymentHealthLink={deploymentHealthLink}
    />
  );
};
