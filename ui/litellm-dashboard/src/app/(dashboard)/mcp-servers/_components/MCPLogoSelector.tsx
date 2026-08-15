import React from "react";
import { Info, Link as LinkIcon } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { cn } from "@/lib/cva.config";
import { Logo } from "@/components/molecules/logo/Logo";
import githubLogo from "../../../../../public/assets/logos/github.svg";
import slackLogo from "../../../../../public/assets/logos/slack.svg";
import notionLogo from "../../../../../public/assets/logos/notion.svg";
import linearLogo from "../../../../../public/assets/logos/linear.svg";
import jiraLogo from "../../../../../public/assets/logos/jira.svg";
import figmaLogo from "../../../../../public/assets/logos/figma.svg";
import gmailLogo from "../../../../../public/assets/logos/gmail.svg";
import googleDriveLogo from "../../../../../public/assets/logos/google_drive.svg";
import stripeLogo from "../../../../../public/assets/logos/stripe.svg";
import shopifyLogo from "../../../../../public/assets/logos/shopify.svg";
import salesforceLogo from "../../../../../public/assets/logos/salesforce.svg";
import hubspotLogo from "../../../../../public/assets/logos/hubspot.svg";
import twilioLogo from "../../../../../public/assets/logos/twilio.svg";
import cloudflareLogo from "../../../../../public/assets/logos/cloudflare.svg";
import sentryLogo from "../../../../../public/assets/logos/sentry.svg";
import postgresqlLogo from "../../../../../public/assets/logos/postgresql.svg";
import snowflakeLogo from "../../../../../public/assets/logos/snowflake.svg";
import zapierLogo from "../../../../../public/assets/logos/zapier.svg";
import googleLogo from "../../../../../public/assets/logos/google.svg";
import gitlabLogo from "../../../../../public/assets/logos/gitlab.svg";

const logos = "/ui/assets/logos/";

const WELL_KNOWN_LOGOS: { name: string; url: string; src: string }[] = [
  { name: "GitHub", url: `${logos}github.svg`, src: githubLogo.src },
  { name: "Slack", url: `${logos}slack.svg`, src: slackLogo.src },
  { name: "Notion", url: `${logos}notion.svg`, src: notionLogo.src },
  { name: "Linear", url: `${logos}linear.svg`, src: linearLogo.src },
  { name: "Jira", url: `${logos}jira.svg`, src: jiraLogo.src },
  { name: "Figma", url: `${logos}figma.svg`, src: figmaLogo.src },
  { name: "Gmail", url: `${logos}gmail.svg`, src: gmailLogo.src },
  { name: "Google Drive", url: `${logos}google_drive.svg`, src: googleDriveLogo.src },
  { name: "Stripe", url: `${logos}stripe.svg`, src: stripeLogo.src },
  { name: "Shopify", url: `${logos}shopify.svg`, src: shopifyLogo.src },
  { name: "Salesforce", url: `${logos}salesforce.svg`, src: salesforceLogo.src },
  { name: "HubSpot", url: `${logos}hubspot.svg`, src: hubspotLogo.src },
  { name: "Twilio", url: `${logos}twilio.svg`, src: twilioLogo.src },
  { name: "Cloudflare", url: `${logos}cloudflare.svg`, src: cloudflareLogo.src },
  { name: "Sentry", url: `${logos}sentry.svg`, src: sentryLogo.src },
  { name: "PostgreSQL", url: `${logos}postgresql.svg`, src: postgresqlLogo.src },
  { name: "Snowflake", url: `${logos}snowflake.svg`, src: snowflakeLogo.src },
  { name: "Zapier", url: `${logos}zapier.svg`, src: zapierLogo.src },
  { name: "Google", url: `${logos}google.svg`, src: googleLogo.src },
  { name: "GitLab", url: `${logos}gitlab.svg`, src: gitlabLogo.src },
];

interface MCPLogoSelectorProps {
  value?: string;
  onChange?: (url: string | undefined) => void;
}

const MCPLogoSelector: React.FC<MCPLogoSelectorProps> = ({ value, onChange }) => {
  const selectedWellKnown = WELL_KNOWN_LOGOS.find((l) => l.url === value);

  const handleSelect = (url: string) => {
    onChange?.(value === url ? undefined : url);
  };

  return (
    <TooltipProvider>
      <div>
        <div className="mb-2 flex items-center gap-2">
          <span className="text-sm font-medium">Logo</span>
          <Tooltip>
            <TooltipTrigger
              render={<Info className="size-4 cursor-help text-muted-foreground" aria-label="About the logo" />}
            />
            <TooltipContent>
              Select a well-known logo or paste a URL to any image. The logo is shown on the admin and chat pages.
            </TooltipContent>
          </Tooltip>
        </div>

        {/* Preview */}
        {value && (
          <div className="mb-3 flex items-center gap-3 rounded-lg border border-border bg-muted p-3">
            <Logo
              src={selectedWellKnown?.src ?? value}
              label="Selected"
              className="h-10 w-10 rounded-sm object-contain"
            />
            <div className="min-w-0 flex-1">
              <div className="truncate text-xs text-muted-foreground">{value}</div>
            </div>
            <button
              type="button"
              onClick={() => onChange?.(undefined)}
              className="cursor-pointer border-none bg-transparent text-xs text-muted-foreground hover:text-destructive"
            >
              ✕
            </button>
          </div>
        )}

        {/* Well-known logo grid */}
        <div className="mb-3 grid grid-cols-10 gap-1.5">
          {WELL_KNOWN_LOGOS.map((logo) => {
            const isSelected = value === logo.url;
            return (
              <Tooltip key={logo.name}>
                <TooltipTrigger
                  render={
                    <button
                      type="button"
                      onClick={() => handleSelect(logo.url)}
                      className={cn(
                        "flex size-10 cursor-pointer items-center justify-center rounded-lg border p-2 transition-all",
                        isSelected ? "border-primary bg-accent shadow-xs" : "border-border hover:bg-accent",
                      )}
                    >
                      <img src={logo.src} alt={logo.name} className="h-5 w-5 object-contain" />
                    </button>
                  }
                />
                <TooltipContent>{logo.name}</TooltipContent>
              </Tooltip>
            );
          })}
        </div>

        {/* Custom URL input */}
        <InputGroup>
          <InputGroupAddon>
            <LinkIcon className="size-4 text-muted-foreground" />
          </InputGroupAddon>
          <InputGroupInput
            placeholder="Or paste a custom logo URL..."
            value={value && !selectedWellKnown ? value : ""}
            onChange={(e) => {
              const v = e.target.value.trim();
              onChange?.(v || undefined);
            }}
          />
        </InputGroup>
      </div>
    </TooltipProvider>
  );
};

export default MCPLogoSelector;
