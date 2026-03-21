import React, { useState } from "react";
import { Input, Tooltip } from "antd";
import { InfoCircleOutlined, LinkOutlined } from "@ant-design/icons";

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
        <span className="text-sm font-medium text-gray-700">Logo</span>
        <Tooltip title="Select a well-known logo or paste a URL to any image. The logo is shown on the admin and chat pages.">
          <InfoCircleOutlined className="text-blue-400 hover:text-blue-600 cursor-help" />
        </Tooltip>
      </div>

      {/* Preview */}
      {value && (
        <div className="flex items-center gap-3 mb-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <img
            src={value}
            alt="Selected logo"
            className="w-10 h-10 object-contain rounded"
            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
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
          const hasFailed = imgErrors.has(logo.url);
          if (hasFailed) return null;
          return (
            <Tooltip key={logo.name} title={logo.name}>
              <button
                type="button"
                onClick={() => handleSelect(logo.url)}
                className={`flex items-center justify-center p-2 rounded-lg border transition-all cursor-pointer
                  ${isSelected
                    ? "border-blue-500 bg-blue-50 shadow-sm"
                    : "border-gray-200 hover:border-blue-300 hover:bg-gray-50"
                  }`}
                style={{ width: 40, height: 40 }}
              >
                <img
                  src={logo.url}
                  alt={logo.name}
                  className="w-5 h-5 object-contain"
                  onError={() => handleImgError(logo.url)}
                />
              </button>
            </Tooltip>
          );
        })}
      </div>

      {/* Custom URL input */}
      <Input
        prefix={<LinkOutlined className="text-gray-400" />}
        placeholder="Or paste a custom logo URL..."
        value={value && !WELL_KNOWN_LOGOS.some((l) => l.url === value) ? value : ""}
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
