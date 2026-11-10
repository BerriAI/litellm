import React from "react";
import { Logo } from "@/components/molecules/logo/Logo";

interface ProviderLogoProps {
  provider: string;
  logoUrl?: string | null;
  className?: string;
}

export const ProviderLogo: React.FC<ProviderLogoProps> = ({ provider, logoUrl, className = "w-4 h-4" }) =>
  logoUrl ? (
    <Logo src={logoUrl} label={provider} className={className} />
  ) : (
    <Logo provider={provider} className={className} />
  );
