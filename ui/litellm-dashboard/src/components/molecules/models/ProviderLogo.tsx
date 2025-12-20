import React, { useState } from "react";
import { getProviderLogoAndName } from "../../provider_info_helpers";

interface ProviderLogoProps {
  provider: string;
  className?: string;
}

export const ProviderLogo: React.FC<ProviderLogoProps> = ({ provider, className = "w-4 h-4" }) => {
  const [hasError, setHasError] = useState(false);
  const { logo } = getProviderLogoAndName(provider);

  const showFallback = hasError || !logo;

  if (showFallback) {
    return (
      <div className={`${className} rounded-full bg-gray-200 dark:bg-zinc-700 flex items-center justify-center text-xs text-gray-600 dark:text-gray-300`}>
        {provider?.charAt(0) || "-"}
      </div>
    );
  }

  return <img src={logo} alt={`${provider} logo`} className={`${className} provider-logo`} onError={() => setHasError(true)} />;
};
