"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import type { Translate } from "@/i18n/language";
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
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/cva.config";
import { isNewerVersion } from "@/utils/versionUtils";
import { Bell } from "lucide-react";
import React, { Fragment } from "react";

export const AUTO_ROUTER_DOCS_URL = "https://docs.litellm.ai/docs/proxy/auto_routing";

interface Announcement {
  readKey: string;
  title: string;
  description: string;
  linkLabel: string;
  href: string;
}

const autoRouterAnnouncement = (t: Translate): Announcement => ({
  readKey: "litellmHideAutoRouterAnnouncement",
  title: t("LiteLLM Auto Router"),
  description: t("Route every request to the cheapest model that can handle it, no prompt changes needed."),
  linkLabel: t("Read the docs"),
  href: AUTO_ROUTER_DOCS_URL,
});

// 只翻译本地构造的发行摘要，发行版本、链接和博客原文保持后端值。
const describeRelease = ({ new_features, bug_fixes, other_updates }: LatestReleaseInfo, t: Translate): string =>
  t("{features}, {fixes}, and {updates}", {
    features: t(new_features === 1 ? "{count} new feature" : "{count} new features", { count: new_features }),
    fixes: t(bug_fixes === 1 ? "{count} fix" : "{count} fixes", { count: bug_fixes }),
    updates: t(other_updates === 1 ? "{count} other update" : "{count} other updates", { count: other_updates }),
  });

const upgradeAnnouncement = (
  currentVersion: string | null | undefined,
  latestRelease: LatestReleaseInfo | null | undefined,
  t: Translate,
): Announcement | null => {
  if (!currentVersion || !latestRelease || !isNewerVersion(currentVersion, latestRelease.version)) {
    return null;
  }
  return {
    readKey: `litellm:upgradeBannerDismissed:${latestRelease.version}`,
    title: t("LiteLLM v{version} is available", { version: latestRelease.version }),
    description: t("{releaseSummary}. Your current version is v{currentVersion}", {
      releaseSummary: describeRelease(latestRelease, t),
      currentVersion,
    }),
    linkLabel: t("View release notes"),
    href: latestRelease.release_url,
  };
};

const AnnouncementItem: React.FC<{ announcement: Announcement }> = ({ announcement }) => {
  const { t } = useLanguage();
  const read = useAnnouncementRead(announcement.readKey);
  return (
    <article aria-label={announcement.title} className="flex flex-col gap-2">
      <p className="text-sm font-medium">{announcement.title}</p>
      <p className="text-sm leading-snug text-muted-foreground">{announcement.description}</p>
      <div className="flex flex-wrap items-center gap-2">
        <a
          className={cn(buttonVariants({ size: "sm" }))}
          href={announcement.href}
          target="_blank"
          rel="noopener noreferrer"
        >
          {announcement.linkLabel}
        </a>
        {read ? null : (
          <Button variant="link" size="sm" className="px-1!" onClick={() => markAnnouncementRead(announcement.readKey)}>
            {t("Mark as read")}
          </Button>
        )}
      </div>
    </article>
  );
};

interface NotificationsBellViewProps {
  currentVersion: string | null | undefined;
  latestRelease: LatestReleaseInfo | null | undefined;
}

export const NotificationsBellView: React.FC<NotificationsBellViewProps> = ({ currentVersion, latestRelease }) => {
  const { t } = useLanguage();
  const upgrade = upgradeAnnouncement(currentVersion, latestRelease, t);
  const autoRouter = autoRouterAnnouncement(t);
  const announcements = upgrade ? [upgrade, autoRouter] : [autoRouter];
  const unreadCount = useUnreadAnnouncementCount(announcements.map((announcement) => announcement.readKey));

  return (
    <Popover>
      <PopoverTrigger
        className="flex! h-9! w-9! items-center justify-center rounded-md! text-muted-foreground transition-colors hover:bg-accent! hover:text-foreground!"
        aria-label={unreadCount > 0 ? t("Notifications, {count} unread", { count: unreadCount }) : t("Notifications")}
      >
        <span className="relative inline-flex">
          <Bell className="size-4" aria-hidden />
          {unreadCount > 0 ? <Badge className="absolute -top-0.5 -right-1 size-1.5 p-0" aria-hidden /> : null}
        </span>
      </PopoverTrigger>
      <PopoverContent align="end" aria-label={t("Notifications")}>
        {announcements.map((announcement, index) => (
          <Fragment key={announcement.readKey}>
            {index > 0 ? <Separator /> : null}
            <AnnouncementItem announcement={announcement} />
          </Fragment>
        ))}
      </PopoverContent>
    </Popover>
  );
};

export const NotificationsBell: React.FC<{ accessToken: string | null }> = ({ accessToken }) => {
  const { data: healthData } = useHealthReadinessDetails(accessToken);
  const { data: latestRelease } = useLatestReleaseInfo(accessToken);
  return <NotificationsBellView currentVersion={healthData?.litellm_version} latestRelease={latestRelease} />;
};
