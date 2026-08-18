import React from "react";
import { Logo } from "@/components/molecules/logo/Logo";
import dataforseoLogo from "../../../../../public/assets/logos/dataforseo.png";
import exaAiLogo from "../../../../../public/assets/logos/exa_ai.png";
import googlePseLogo from "../../../../../public/assets/logos/google_pse.png";
import nimbleLogo from "../../../../../public/assets/logos/nimble.png";
import parallelAiLogo from "../../../../../public/assets/logos/parallel_ai.png";
import perplexityLogo from "../../../../../public/assets/logos/perplexity.png";
import tavilyLogo from "../../../../../public/assets/logos/tavily.png";

const searchProviderLogoMap: Readonly<Record<string, string>> = {
  perplexity: perplexityLogo.src,
  tavily: tavilyLogo.src,
  parallel_ai: parallelAiLogo.src,
  exa_ai: exaAiLogo.src,
  google_pse: googlePseLogo.src,
  dataforseo: dataforseoLogo.src,
  nimble: nimbleLogo.src,
};

interface SearchProviderLabelProps {
  providerName: string;
  displayName: string;
  className?: string;
  logoClassName?: string;
}

export const SearchProviderLabel: React.FC<SearchProviderLabelProps> = ({
  providerName,
  displayName,
  className = "text-sm",
  logoClassName = "w-5 h-5 object-contain",
}) => (
  <div className={`flex items-center gap-2 ${className}`}>
    <Logo src={searchProviderLogoMap[providerName]} label={displayName} className={logoClassName} />
    <span>{displayName}</span>
  </div>
);
