"use client";

import type { ParseKeys } from "i18next";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { BookOpen, Github, Newspaper, ScrollText, Slack } from "lucide-react";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { useBlogPosts } from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";
import { Button, buttonVariants } from "@/components/ui/button";
import { ToolbarSeparator } from "@/components/shared/ToolbarSeparator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";

function DiscordIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M20.32 4.37a19.8 19.8 0 0 0-4.89-1.52.07.07 0 0 0-.08.04c-.21.38-.44.87-.61 1.25a18.27 18.27 0 0 0-5.49 0 12.6 12.6 0 0 0-.62-1.25.08.08 0 0 0-.08-.04 19.74 19.74 0 0 0-4.89 1.52.07.07 0 0 0-.03.03C.53 9.05-.32 13.58.1 18.06a.08.08 0 0 0 .03.06 19.9 19.9 0 0 0 5.99 3.03.08.08 0 0 0 .08-.03c.46-.63.87-1.3 1.23-1.99a.08.08 0 0 0-.04-.11 13.1 13.1 0 0 1-1.87-.89.08.08 0 0 1-.01-.13l.37-.29a.07.07 0 0 1 .08-.01c3.93 1.79 8.18 1.79 12.06 0a.07.07 0 0 1 .08.01l.37.29a.08.08 0 0 1-.01.13c-.6.35-1.22.65-1.87.89a.08.08 0 0 0-.04.11c.36.7.78 1.36 1.23 1.99a.08.08 0 0 0 .08.03 19.84 19.84 0 0 0 6-3.03.08.08 0 0 0 .03-.05c.5-5.18-.84-9.67-3.55-13.66a.06.06 0 0 0-.03-.03ZM8.02 15.33c-1.18 0-2.16-1.08-2.16-2.42 0-1.33.96-2.42 2.16-2.42 1.21 0 2.18 1.1 2.16 2.42 0 1.34-.96 2.42-2.16 2.42Zm7.97 0c-1.18 0-2.15-1.08-2.15-2.42 0-1.33.95-2.42 2.15-2.42 1.21 0 2.18 1.1 2.16 2.42 0 1.34-.95 2.42-2.16 2.42Z" />
    </svg>
  );
}

const COMMUNITY = [
  {
    href: "https://www.litellm.ai/support",
    label: "Slack",
    accessibleLabel: "shell.dashboardCommunityLinks.joinSlack",
    Icon: Slack,
  },
  {
    href: "https://discord.gg/wuPM9dRgDw",
    label: "Discord",
    accessibleLabel: "shell.dashboardCommunityLinks.joinDiscord",
    Icon: DiscordIcon,
  },
  {
    href: "https://github.com/BerriAI/litellm",
    label: "GitHub",
    accessibleLabel: "shell.dashboardCommunityLinks.github",
    Icon: Github,
  },
] as const;
type ResourceLink = { href: string; label: ParseKeys; Icon: typeof BookOpen };

const DOCS: readonly ResourceLink[] = [
  { href: "https://docs.litellm.ai/docs/", label: "shell.dashboardResources.links.docs", Icon: BookOpen },
  { href: "https://docs.litellm.ai/release_notes", label: "shell.dashboardResources.links.releases", Icon: ScrollText },
];
const BLOG: ResourceLink = {
  href: "https://docs.litellm.ai/blog",
  label: "shell.dashboardResources.links.blog",
  Icon: Newspaper,
};

export function DashboardCommunityLinks({ className }: { className?: string }) {
  const { t } = useTranslation();
  const hidden = useDisableShowPrompts();
  if (hidden) return null;
  return (
    <nav aria-label={t("shell.dashboardCommunityLinks.label")} className={cn("items-center gap-0.5", className)}>
      <TooltipProvider>
        {COMMUNITY.map(({ href, label, accessibleLabel, Icon }) => (
          <Tooltip key={label}>
            <TooltipTrigger
              render={
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={t(accessibleLabel)}
                  className={cn(buttonVariants({ variant: "ghost", size: "icon" }), "text-muted-foreground")}
                />
              }
            >
              <Icon />
            </TooltipTrigger>
            <TooltipContent>{label}</TooltipContent>
          </Tooltip>
        ))}
      </TooltipProvider>
      <ToolbarSeparator className="mx-1.5 h-4 bg-foreground/15" />
    </nav>
  );
}

export function DashboardResources({ children }: { children?: ReactNode }) {
  const { t } = useTranslation();
  const hideBlog = useDisableBlogPosts();
  const hideCommunity = useDisableShowPrompts();
  const { data, isPending, isError, refetch } = useBlogPosts();
  const links = [
    ...[...DOCS, ...(hideBlog ? [] : [BLOG])].map((link) => ({ ...link, label: t(link.label) })),
    ...(hideCommunity ? [] : COMMUNITY),
  ];
  return (
    <>
      {!hideBlog && (
        <section aria-label={t("shell.dashboardResources.latestPosts")} className="border-t border-border px-2 py-2">
          {isPending && (
            <p className="px-2 py-3 text-xs text-muted-foreground">{t("shell.dashboardResources.loadingPosts")}</p>
          )}
          {isError && (
            <Button variant="ghost" size="sm" onClick={() => void refetch()}>
              {t("shell.dashboardResources.retryPosts")}
            </Button>
          )}
          {data?.posts.slice(0, 3).map((post) => (
            <a
              key={post.url}
              href={post.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-baseline justify-between gap-4 rounded-md px-2 py-2 text-[13px] leading-5 hover:bg-muted focus-visible:outline-ring"
            >
              <span className="line-clamp-2">{post.title}</span>
              <time dateTime={post.date} className="shrink-0 text-[11px] text-muted-foreground">
                {new Date(`${post.date}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
              </time>
            </a>
          ))}
        </section>
      )}
      <div className="sticky bottom-0 border-t border-border bg-popover p-2">
        <nav aria-label={t("shell.dashboardResources.helpAndResources")} className="grid grid-cols-3 gap-1">
          {links.map(({ href, label, Icon }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                buttonVariants({ variant: "ghost", size: "sm" }),
                "justify-start gap-2 text-xs font-normal text-muted-foreground",
              )}
            >
              <Icon />
              {label}
            </a>
          ))}
        </nav>
        {children}
      </div>
    </>
  );
}
