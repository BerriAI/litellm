import React from "react";
import { Logo } from "@/components/molecules/logo/Logo";

interface ProviderLogoProps {
  provider: string;
  className?: string;
}

export const ProviderLogo: React.FC<ProviderLogoProps> = ({ provider, className = "w-4 h-4" }) => (
  <Logo provider={provider} className={className} />
);
