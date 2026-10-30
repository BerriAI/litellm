import React, { useState } from "react";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { resolveLogoSrc } from "@/lib/assetPaths";
import { cn } from "@/lib/cva.config";
import { logoTreatmentFor, type LogoTreatment } from "@/lib/logoTreatments";

type LogoProps = { className?: string } & (
  | { provider: string; src?: never; label?: string }
  | { provider?: never; src: string | null | undefined; label: string }
);

const DARK_TREATMENT_CLASS: Readonly<Record<LogoTreatment, string>> = {
  invert: "dark:[filter:brightness(0)_invert(1)]",
  plate: "dark:bg-logo-surface dark:object-contain dark:p-0.5",
};

export const Logo: React.FC<LogoProps> = ({ provider, src, label, className = "w-4 h-4" }) => {
  const [erroredSrc, setErroredSrc] = useState<string | null>(null);
  const resolvedSrc = provider !== undefined ? getProviderLogoAndName(provider).logo : resolveLogoSrc(src) ?? "";
  const name = label ?? provider ?? "";

  if (erroredSrc === resolvedSrc || !resolvedSrc) {
    return (
      <div className={`${className} rounded-full bg-border flex items-center justify-center text-xs`}>
        {name.charAt(0) || "-"}
      </div>
    );
  }

  const treatment = logoTreatmentFor(resolvedSrc);

  return (
    <img
      src={resolvedSrc}
      alt={`${name || "-"} logo`}
      className={treatment === undefined ? className : cn(className, DARK_TREATMENT_CLASS[treatment])}
      onError={() => {
        console.warn(`Logo failed to load: ${resolvedSrc}`);
        setErroredSrc(resolvedSrc);
      }}
    />
  );
};
