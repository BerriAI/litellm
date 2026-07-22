import React from "react";
import { Input, Tooltip } from "antd";
import { InfoCircleOutlined, LinkOutlined } from "@ant-design/icons";
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
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-medium text-gray-700">Logo</span>
        <Tooltip title="Select a well-known logo or paste a URL to any image. The logo is shown on the admin and chat pages.">
          <InfoCircleOutlined className="text-blue-400 hover:text-blue-600 cursor-help" />
        </Tooltip>
      </div>

      {/* Preview */}
      {value && (
        <div className="flex items-center gap-3 mb-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <Logo
            src={selectedWellKnown?.src ?? value}
            label="Selected"
            className="w-10 h-10 object-contain rounded-sm"
          />
          <div className="flex-1 min-w-0">
            <div className="text-xs text-gray-500 truncate">{value}</div>
          </div>
          <button
            type="button"
            onClick={() => onChange?.(undefined)}
            className="text-xs text-gray-400 hover:text-red-500 cursor-pointer bg-transparent border-none"
          >
            ✕
          </button>
        </div>
      )}

      {/* Well-known logo grid */}
      <div className="grid grid-cols-10 gap-1.5 mb-3">
        {WELL_KNOWN_LOGOS.map((logo) => {
          const isSelected = value === logo.url;
          return (
            <Tooltip key={logo.name} title={logo.name}>
              <button
                type="button"
                onClick={() => handleSelect(logo.url)}
                className={`flex items-center justify-center p-2 rounded-lg border transition-all cursor-pointer
                  ${
                    isSelected
                      ? "border-blue-500 bg-blue-50 shadow-xs"
                      : "border-gray-200 hover:border-blue-300 hover:bg-gray-50"
                  }`}
                style={{ width: 40, height: 40 }}
              >
                <img src={logo.src} alt={logo.name} className="w-5 h-5 object-contain" />
              </button>
            </Tooltip>
          );
        })}
      </div>

      {/* Custom URL input */}
      <Input
        prefix={<LinkOutlined className="text-gray-400" />}
        placeholder="Or paste a custom logo URL..."
        value={value && !selectedWellKnown ? value : ""}
        onChange={(e) => {
          const v = e.target.value.trim();
          onChange?.(v || undefined);
        }}
        className="rounded-lg"
        size="small"
      />
    </div>
  );
};

export default MCPLogoSelector;
