import React from "react";
import { Logo } from "@/components/molecules/logo/Logo";
import { getModelLogo } from "@/components/provider_info_helpers";

interface ModelLogoProps {
  model: string | undefined;
  provider: string;
  className?: string;
}

export const ModelLogo: React.FC<ModelLogoProps> = ({ model, provider, className = "w-4 h-4" }) => (
  <Logo src={getModelLogo(model, provider)} label={provider} className={className} />
);
