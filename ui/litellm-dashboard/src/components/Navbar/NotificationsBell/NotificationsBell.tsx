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

const AUTO_ROUTER_ANNOUNCEMENT: Announcement = {
  readKey: "litellmHideAutoRouterAnnouncement",
  title: "LiteLLM Auto Router",
  description: "Route every request to the cheapest model that can handle it, no prompt changes needed.",
  linkLabel: "Read the docs",
  href: AUTO_ROUTER_DOCS_URL,
};

const plural = (count: number, singular: string, pluralForm: string): string =>
  `${count} ${count === 1 ? singular : pluralForm}`;

const describeRelease = ({ new_features, bug_fixes, other_updates }: LatestReleaseInfo): string =>
  [
    plural(new_features, "new feature", "new features"),
    plural(bug_fixes, "fix", "fixes"),
    `and ${plural(other_updates, "other update", "other updates")}`,
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
    title: `LiteLLM v${latestRelease.version} is available`,
    description: `${describeRelease(latestRelease)}. Your current version is v${currentVersion}`,
    linkLabel: "View release notes",
    href: latestRelease.release_url,
  };
};

const AnnouncementItem: React.FC<{ announcement: Announcement }> = ({ announcement }) => {
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
            Mark as read
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
  const upgrade = upgradeAnnouncement(currentVersion, latestRelease);
  const announcements = upgrade ? [upgrade, AUTO_ROUTER_ANNOUNCEMENT] : [AUTO_ROUTER_ANNOUNCEMENT];
  const unreadCount = useUnreadAnnouncementCount(announcements.map((announcement) => announcement.readKey));

  return (
    <Popover>
      <PopoverTrigger
        className="flex! h-9! w-9! items-center justify-center rounded-md! text-muted-foreground transition-colors hover:bg-accent! hover:text-foreground!"
        aria-label={unreadCount > 0 ? `Notifications, ${unreadCount} unread` : "Notifications"}
      >
        <span className="relative inline-flex">
          <Bell className="size-4" aria-hidden />
          {unreadCount > 0 ? <Badge className="absolute -top-0.5 -right-1 size-1.5 p-0" aria-hidden /> : null}
        </span>
      </PopoverTrigger>
      <PopoverContent align="end" aria-label="Notifications">
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
