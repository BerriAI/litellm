"use client";

import { MessagesSquare } from "lucide-react";

import githubLogo from "../../../../../public/assets/logos/github.svg";
import jiraLogo from "../../../../../public/assets/logos/jira.svg";
import linearLogo from "../../../../../public/assets/logos/linear.svg";
import slackLogo from "../../../../../public/assets/logos/slack.svg";
import { Logo } from "@/components/molecules/logo/Logo";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";

import type { TraceSummary } from "../types";

type Source = NonNullable<TraceSummary["source"]>;
type SourceType = Source["type"];

interface SourceApp {
  readonly label: string;
  readonly link: string;
  readonly logo: string | null;
}

const APPS: Readonly<Record<SourceType, SourceApp>> = {
  slack: { label: "Slack", link: "Slack thread", logo: slackLogo.src },
  teams: { label: "Microsoft Teams", link: "Teams thread", logo: null },
  discord: { label: "Discord", link: "Discord thread", logo: null },
  linear: { label: "Linear", link: "Linear issue", logo: linearLogo.src },
  github: { label: "GitHub", link: "GitHub thread", logo: githubLogo.src },
  jira: { label: "Jira", link: "Jira issue", logo: jiraLogo.src },
  custom: { label: "Conversation", link: "Agent conversation", logo: null },
};

export function sourceApp(source: Pick<Source, "type" | "url">): SourceApp | null {
  const parsed = URL.canParse(source.url) ? new URL(source.url) : null;
  if (parsed?.protocol !== "https:") return null;
  return APPS[source.type] ?? APPS.custom;
}

function AppMark({ app, className }: { app: SourceApp; className: string }) {
  return app.logo ? (
    <Logo src={app.logo} label={app.label} className={className} />
  ) : (
    <MessagesSquare aria-hidden className={className} />
  );
}

/** Links a run back to the conversation that started it, e.g. a Slack thread. */
export function RunSourceLink({ source }: { source: Source }) {
  const app = sourceApp(source);
  if (!app) return null;
  return (
    <HoverCard>
      <HoverCardTrigger
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-muted/60 px-2 py-0.5 font-medium text-foreground transition-colors hover:bg-muted active:scale-[0.97]"
      >
        <AppMark app={app} className="size-3 shrink-0" />
        {app.link}
      </HoverCardTrigger>
      <HoverCardContent align="start" className="w-72 p-3">
        <a href={source.url} target="_blank" rel="noopener noreferrer" className="flex flex-col gap-1.5">
          <span className="line-clamp-2 text-sm font-medium text-foreground">
            {source.title || `Open in ${app.label}`}
          </span>
          <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <AppMark app={app} className="size-3 shrink-0" />
            {app.label}
          </span>
        </a>
      </HoverCardContent>
    </HoverCard>
  );
}
