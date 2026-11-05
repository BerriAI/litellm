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
import { DashboardBlogPosts, DashboardResources } from "@/components/Navbar/DashboardResources";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/cva.config";
import { isNewerVersion } from "@/utils/versionUtils";
import { AlertTriangle, ArrowUpRight, Bell, ChevronDown } from "lucide-react";
import DeprecationAlertItem from "@/components/ModelDeprecationAlerts/DeprecationAlertItem";
import { type DeprecationAlert } from "@/components/ModelDeprecationAlerts/deprecationAlerts";
import { useDeprecationAlerts } from "@/components/ModelDeprecationAlerts/useDeprecationAlerts";
import { AlertStatus, effectiveAlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { migrationHref } from "@/components/ModelDeprecationAlerts/migrationRoute";
import {
  type AlertGroupPage,
  formatAlertChange,
  METRIC_SHORT_LABELS,
  SCOPE_LABELS,
  usageAlertGroupName,
  usageAlertGroupHref,
  useUsageAlertGroups,
} from "@/components/UsageAlerts/usageAlertApi";
import { uiHref } from "@/utils/uiHref";
import React, { Fragment, useState } from "react";
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

const AUTO_ROUTER_ANNOUNCEMENT: Announcement = {
  readKey: "litellmHideAutoRouterAnnouncement",
  title: "LiteLLM Auto Router",
  description: "Route every request to the cheapest model that can handle it, no prompt changes needed.",
  linkLabel: "Read the docs",
  href: AUTO_ROUTER_DOCS_URL,
};

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

function ProductUpdates({
  announcements,
  unreadCount,
  hasAlerts,
}: {
  announcements: Announcement[];
  unreadCount: number;
  hasAlerts: boolean;
}) {
  const [open, setOpen] = useState<boolean | undefined>();
  const expanded = open ?? !hasAlerts;
  return (
    <Collapsible open={expanded} onOpenChange={setOpen} className="shrink-0 border-t border-border px-4 py-2">
      <CollapsibleTrigger className="flex w-full items-center justify-between gap-2 py-1 text-sm font-medium">
        <span>What’s new{unreadCount > 0 ? ` · ${unreadCount} unread` : ""}</span>
        <ChevronDown
          className={cn("size-4 text-muted-foreground transition-transform", expanded && "rotate-180")}
          aria-hidden
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="max-h-[min(14rem,25dvh)] overflow-y-auto overscroll-contain">
        <div className="space-y-3 pt-3">
          {announcements.map((announcement, index) => (
            <Fragment key={announcement.readKey}>
              {index > 0 ? <Separator /> : null}
              <AnnouncementItem announcement={announcement} />
            </Fragment>
          ))}
          <DashboardBlogPosts />
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

interface NotificationsBellViewProps {
  deprecationAlerts?: readonly DeprecationAlert[];
  usageAlertGroups?: AlertGroupPage;
  deploymentHealthLink?: React.ReactNode;
  currentVersion: string | null | undefined;
  latestRelease: LatestReleaseInfo | null | undefined;
}

export const NotificationsBellView: React.FC<NotificationsBellViewProps> = ({
  currentVersion,
  latestRelease,
  deploymentHealthLink,
  deprecationAlerts,
  usageAlertGroups,
}) => {
  const { t } = useTranslation();
  const upgrade = upgradeAnnouncement(currentVersion, latestRelease);
  const announcements = upgrade ? [upgrade, AUTO_ROUTER_ANNOUNCEMENT] : [AUTO_ROUTER_ANNOUNCEMENT];
  const unreadCount = useUnreadAnnouncementCount(announcements.map((announcement) => announcement.readKey));

  const openAlerts = (deprecationAlerts ?? []).filter((alert) => effectiveAlertStatus(alert) === AlertStatus.Open);
  const openDeprecations = openAlerts.length;
  const openUsageAlerts = usageAlertGroups?.total ?? 0;
  const needsAttention = openDeprecations + openUsageAlerts > 0;
  let label =
    unreadCount > 0
      ? t("shell.notifications.bell.triggerUnread", { unread: unreadCount })
      : t("shell.notifications.bell.trigger");
  if (openDeprecations > 0) {
    label = `Notifications, ${openDeprecations} model deprecation ${openDeprecations === 1 ? "alert needs" : "alerts need"} attention`;
  }
  if (openUsageAlerts > 0) {
    const count = openDeprecations + openUsageAlerts;
    label = `Notifications, ${count} ${count === 1 ? "alert group needs" : "alert groups need"} attention`;
  }

  return (
    <Popover>
      <PopoverTrigger
        render={<Button variant="ghost" size="icon" className="text-muted-foreground" />}
        aria-label={label}
      >
        <span className="relative inline-flex">
          <Bell className={cn("size-4", needsAttention && "text-amber-700")} aria-hidden />
          {needsAttention ? (
            <span className="absolute -top-2 -right-2 rounded-full bg-background p-px text-amber-600" aria-hidden>
              <AlertTriangle className="size-3" fill="currentColor" stroke="var(--background)" strokeWidth={1.5} />
            </span>
          ) : null}
          {!needsAttention && unreadCount > 0 ? (
            <Badge className="absolute -top-0.5 -right-1 size-1.5 p-0" aria-hidden />
          ) : null}
        </span>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        aria-label={t("shell.notifications.bell.trigger")}
        className="flex max-h-[min(40rem,calc(100dvh-6rem))] w-96 max-w-[calc(100vw-1rem)] flex-col gap-0 overflow-hidden p-0"
      >
        <PopoverTitle className="shrink-0 border-b border-border px-4 py-3 text-sm font-semibold">
          {t("shell.notifications.bell.trigger")}
        </PopoverTitle>
        <div aria-label="Notification content" className="min-h-0 overflow-y-auto overscroll-contain">
          <div className="space-y-3 p-4">
            {openUsageAlerts > 0 && (
              <>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-medium">Performance</p>
                  <a className="text-xs text-primary hover:underline" href={uiHref("alerts?type=usage")}>
                    View all {openUsageAlerts}
                  </a>
                </div>
                {usageAlertGroups?.data.slice(0, 3).map((group) => (
                  <a
                    key={group.id}
                    href={usageAlertGroupHref(group)}
                    className="block space-y-1 rounded-md py-1 text-sm hover:underline"
                  >
                    <p
                      className="truncate font-medium"
                      title={group.scope_label ?? group.scope_id ?? usageAlertGroupName(group)}
                    >
                      {usageAlertGroupName(group)}
                    </p>
                    <p className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
                      {group.metrics.map(({ metric, representative }) => (
                        <span key={metric}>
                          {METRIC_SHORT_LABELS[metric]} {formatAlertChange(representative)}
                        </span>
                      ))}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {SCOPE_LABELS[group.scope_type]} · {group.day} UTC
                    </p>
                  </a>
                ))}
                <Separator />
              </>
            )}
            {openAlerts.length > 0 && (
              <>
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-medium">Model deprecations</p>
                  <a className="text-xs text-primary hover:underline" href={migrationHref()}>
                    View all {openDeprecations}
                  </a>
                </div>
                {openAlerts.slice(0, 3).map((alert) => (
                  <Fragment key={alert.id}>
                    <Separator />
                    <DeprecationAlertItem alert={alert} />
                  </Fragment>
                ))}
                <Separator />
              </>
            )}
            {deprecationAlerts && openAlerts.length === 0 && (
              <a href={migrationHref()} className="text-sm font-medium text-primary hover:underline">
                View model deprecations
              </a>
            )}
          </div>
        </div>
        <ProductUpdates announcements={announcements} unreadCount={unreadCount} hasAlerts={needsAttention} />
        <DashboardResources>{deploymentHealthLink}</DashboardResources>
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
  const { data: usageAlertGroups } = useUsageAlertGroups(accessToken, "open", { limit: 3 });
  return (
    <NotificationsBellView
      currentVersion={healthData?.litellm_version}
      latestRelease={latestRelease}
      deprecationAlerts={deprecationAlerts}
      usageAlertGroups={usageAlertGroups}
      deploymentHealthLink={deploymentHealthLink}
    />
  );
};
