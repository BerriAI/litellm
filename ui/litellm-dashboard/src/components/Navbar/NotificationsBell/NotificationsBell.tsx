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
import DeprecationAlertItem from "@/components/ModelDeprecationAlerts/DeprecationAlertItem";
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

const describeRelease = ({ new_features, bug_fixes, other_updates }: LatestReleaseInfo): string =>
  [
    i18n.t("shell.notifications.release.newFeatures", { count: new_features }),
    i18n.t("shell.notifications.release.fixes", { count: bug_fixes }),
    i18n.t("shell.notifications.release.otherUpdates", { count: other_updates }),
  ].join(", ");

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
    description: i18n.t("shell.notifications.upgrade.description", {
      summary: describeRelease(latestRelease),
      current: currentVersion,
    }),
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
  deploymentHealthLink?: React.ReactNode;
  currentVersion: string | null | undefined;
  latestRelease: LatestReleaseInfo | null | undefined;
}

export const NotificationsBellView: React.FC<NotificationsBellViewProps> = ({
  currentVersion,
  latestRelease,
  deploymentHealthLink,
  deprecationAlerts,
}) => {
  const { t } = useTranslation();
  const upgrade = upgradeAnnouncement(currentVersion, latestRelease);
  const announcements = upgrade ? [upgrade, autoRouterAnnouncement()] : [autoRouterAnnouncement()];
  const unreadCount = useUnreadAnnouncementCount(announcements.map((announcement) => announcement.readKey));

  const openAlerts = (deprecationAlerts ?? []).filter((alert) => effectiveAlertStatus(alert) === AlertStatus.Open);
  const openDeprecations = openAlerts.length;
  let label =
    unreadCount > 0
      ? t("shell.notifications.bell.triggerUnread", { unread: unreadCount })
      : t("shell.notifications.bell.trigger");
  if (openDeprecations > 0) {
    label = t("shell.notifications.bell.triggerDeprecations", { count: openDeprecations });
  }

  return (
    <Popover>
      <PopoverTrigger
        render={<Button variant="ghost" size="icon" className="text-muted-foreground" />}
        aria-label={label}
      >
        <span className="relative inline-flex">
          <Bell className={cn("size-4", openDeprecations > 0 && "text-amber-700")} aria-hidden />
          {openDeprecations > 0 ? (
            <span className="absolute -top-2 -right-2 rounded-full bg-background p-px text-amber-600" aria-hidden>
              <AlertTriangle className="size-3" fill="currentColor" stroke="var(--background)" strokeWidth={1.5} />
            </span>
          ) : null}
          {openDeprecations === 0 && unreadCount > 0 ? (
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
            {openAlerts.length > 0 && (
              <>
                <div>
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold">{t("shell.notifications.bell.deprecationsTitle")}</p>
                    <a className="text-xs text-primary hover:underline" href={migrationHref()}>
                      {t("shell.notifications.bell.viewAllDeprecations", { total: openDeprecations })}
                    </a>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {t("shell.notifications.bell.deprecationsDescription")}
                  </p>
                </div>
                {openAlerts.slice(0, 3).map((alert) => (
                  <Fragment key={alert.id}>
                    <Separator />
                    <DeprecationAlertItem alert={alert} />
                  </Fragment>
                ))}
                <Separator />
                <p className="text-xs font-medium text-muted-foreground">
                  {t("shell.notifications.bell.announcementsHeading")}
                </p>
              </>
            )}
            {deprecationAlerts && openAlerts.length === 0 && (
              <a href={migrationHref()} className="text-sm font-medium text-primary hover:underline">
                {t("shell.notifications.bell.viewDeprecations")}
              </a>
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
  const { data: deprecationAlerts } = useDeprecationAlerts(accessToken);
  return (
    <NotificationsBellView
      currentVersion={healthData?.litellm_version}
      latestRelease={latestRelease}
      deprecationAlerts={deprecationAlerts}
      deploymentHealthLink={deploymentHealthLink}
    />
  );
};
