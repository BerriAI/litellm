import React, { useState } from "react";
import { Input, Tooltip } from "antd";
import { InfoCircleOutlined, LinkOutlined } from "@ant-design/icons";
import { WELL_KNOWN_LOGOS } from "./utils";

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
