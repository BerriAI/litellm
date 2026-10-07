"use client";

import { ExternalLink } from "lucide-react";

import githubLogo from "../../../../../public/assets/logos/github.svg";
import linearLogo from "../../../../../public/assets/logos/linear.svg";
import notionLogo from "../../../../../public/assets/logos/notion.svg";
import slackLogo from "../../../../../public/assets/logos/slack.svg";
import { Logo } from "@/components/molecules/logo/Logo";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";

import type { TraceSummary } from "../types";

type Source = NonNullable<TraceSummary["source"]>;

interface SourceApp {
  readonly label: string;
  readonly logo: string | null;
}

const APPS: readonly (SourceApp & { readonly host: RegExp })[] = [
  { host: /(^|\.)slack\.com$/, label: "Slack", logo: slackLogo.src },
  { host: /(^|\.)linear\.app$/, label: "Linear", logo: linearLogo.src },
  { host: /(^|\.)github\.com$/, label: "GitHub", logo: githubLogo.src },
  { host: /(^|\.)notion\.(so|site)$/, label: "Notion", logo: notionLogo.src },
];

export function sourceApp(url: string): SourceApp | null {
  const parsed = URL.canParse(url) ? new URL(url) : null;
  if (parsed?.protocol !== "https:") return null;
  return APPS.find((app) => app.host.test(parsed.hostname)) ?? { label: parsed.hostname, logo: null };
}

function AppMark({ app, className }: { app: SourceApp; className: string }) {
  return app.logo ? (
    <Logo src={app.logo} label={app.label} className={className} />
  ) : (
    <ExternalLink aria-hidden className={className} />
  );
}

/** Links a run back to where it started, e.g. the Slack thread that asked for it. */
export function RunSourceLink({ source }: { source: Source }) {
  const app = sourceApp(source.url);
  if (!app) return null;
  const title = source.title || `Open in ${app.label}`;
  return (
    <HoverCard>
      <HoverCardTrigger
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`Open source in ${app.label}`}
        className="inline-flex size-7 shrink-0 items-center justify-center rounded-md bg-muted/60 transition-colors hover:bg-muted active:scale-[0.97]"
      >
        <AppMark app={app} className="size-4 shrink-0" />
      </HoverCardTrigger>
      <HoverCardContent align="start" className="w-72 p-3">
        <a href={source.url} target="_blank" rel="noopener noreferrer" className="flex flex-col gap-1.5">
          <span className="line-clamp-2 text-sm font-medium text-foreground">{title}</span>
          <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <AppMark app={app} className="size-3 shrink-0" />
            {app.label}
          </span>
        </a>
      </HoverCardContent>
    </HoverCard>
  );
}
