import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info, Link as LinkIcon } from "lucide-react";
import { cn } from "@/lib/utils";

const logos = "/ui/assets/logos/";

const WELL_KNOWN_LOGOS: { name: string; url: string }[] = [
  { name: "GitHub", url: `${logos}github.svg` },
  { name: "Slack", url: `${logos}slack.svg` },
  { name: "Notion", url: `${logos}notion.svg` },
  { name: "Linear", url: `${logos}linear.svg` },
  { name: "Jira", url: `${logos}jira.svg` },
  { name: "Figma", url: `${logos}figma.svg` },
  { name: "Gmail", url: `${logos}gmail.svg` },
  { name: "Google Drive", url: `${logos}google_drive.svg` },
  { name: "Stripe", url: `${logos}stripe.svg` },
  { name: "Shopify", url: `${logos}shopify.svg` },
  { name: "Salesforce", url: `${logos}salesforce.svg` },
  { name: "HubSpot", url: `${logos}hubspot.svg` },
  { name: "Twilio", url: `${logos}twilio.svg` },
  { name: "Cloudflare", url: `${logos}cloudflare.svg` },
  { name: "Sentry", url: `${logos}sentry.svg` },
  { name: "PostgreSQL", url: `${logos}postgresql.svg` },
  { name: "Snowflake", url: `${logos}snowflake.svg` },
  { name: "Zapier", url: `${logos}zapier.svg` },
  { name: "Google", url: `${logos}google.svg` },
  { name: "GitLab", url: `${logos}gitlab.svg` },
];

interface MCPLogoSelectorProps {
  value?: string;
  onChange?: (url: string | undefined) => void;
}

const MCPLogoSelector: React.FC<MCPLogoSelectorProps> = ({ value, onChange }) => {
  const [imgErrors, setImgErrors] = useState<Set<string>>(new Set());

  const handleSelect = (url: string) => {
    onChange?.(value === url ? undefined : url);
  };

  const handleImgError = (url: string) => {
    setImgErrors((prev) => new Set(prev).add(url));
  };

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-medium text-foreground">Logo</span>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-3 w-3 text-primary cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              Select a well-known logo or paste a URL to any image. The logo is
              shown on the admin and chat pages.
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {value && (
        <div className="flex items-center gap-3 mb-3 p-3 bg-muted rounded-lg border border-border">
          <img
            src={value}
            alt="Selected logo"
            className="w-10 h-10 object-contain rounded"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          <div className="flex-1 min-w-0">
            <div className="text-xs text-muted-foreground truncate">
              {value}
            </div>
          </div>
          <button
            type="button"
            onClick={() => onChange?.(undefined)}
            className="text-xs text-muted-foreground hover:text-destructive cursor-pointer bg-transparent border-none"
            aria-label="Clear selected logo"
          >
            ✕
          </button>
        </div>
      )}

      <div className="grid grid-cols-10 gap-1.5 mb-3">
        {WELL_KNOWN_LOGOS.map((logo) => {
          const isSelected = value === logo.url;
          if (imgErrors.has(logo.url)) return null;
          return (
            <TooltipProvider key={logo.name}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => handleSelect(logo.url)}
                    className={cn(
                      "flex items-center justify-center p-2 rounded-lg border transition-all cursor-pointer",
                      isSelected
                        ? "border-primary bg-primary/10 shadow-sm"
                        : "border-border hover:border-primary/50 hover:bg-muted",
                    )}
                    style={{ width: 40, height: 40 }}
                    aria-label={logo.name}
                  >
                    <img
                      src={logo.url}
                      alt={logo.name}
                      className="w-5 h-5 object-contain"
                      onError={() => handleImgError(logo.url)}
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>{logo.name}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          );
        })}
      </div>

      <div className="relative">
        <LinkIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
        <Input
          placeholder="Or paste a custom logo URL..."
          value={
            value && !WELL_KNOWN_LOGOS.some((l) => l.url === value) ? value : ""
          }
          onChange={(e) => {
            const v = e.target.value.trim();
            onChange?.(v || undefined);
          }}
          className="pl-8 h-8 rounded-lg"
        />
      </div>
    </div>
  );
};

export default MCPLogoSelector;
