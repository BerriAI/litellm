import React, { useState } from "react";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { resolveLogoSrc } from "@/lib/assetPaths";

interface LogoProps {
  provider?: string;
  src?: string | null;
  label?: string;
  className?: string;
}

export const Logo: React.FC<LogoProps> = ({ provider, src, label, className = "w-4 h-4" }) => {
  const [erroredSrc, setErroredSrc] = useState<string | null>(null);
  const resolvedSrc = provider !== undefined ? getProviderLogoAndName(provider).logo : resolveLogoSrc(src) ?? "";
  const name = label ?? provider ?? "";

  if (erroredSrc === resolvedSrc || !resolvedSrc) {
    return (
      <div className={`${className} rounded-full bg-gray-200 flex items-center justify-center text-xs`}>
        {name.charAt(0) || "-"}
      </div>
    );
  }

  return (
    <img
      src={resolvedSrc}
      alt={`${name || "-"} logo`}
      className={className}
      onError={() => {
        console.warn(`Logo failed to load: ${resolvedSrc}`);
        setErroredSrc(resolvedSrc);
      }}
    />
  );
};
