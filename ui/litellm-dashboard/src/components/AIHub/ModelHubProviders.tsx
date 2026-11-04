"use client";

import Image from "next/image";
import { useState } from "react";

import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/cva.config";
import { displayMode } from "@/lib/decisionModels";

export function formatModelModeLabel(mode: string | undefined): string {
  const label = (displayMode(mode) ?? "chat").replaceAll("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

interface ProviderLogoProps {
  provider: string;
  className?: string;
}

export function ProviderLogo({ provider, className }: ProviderLogoProps) {
  const { logo, displayName } = getProviderLogoAndName(provider);
  const [failedLogo, setFailedLogo] = useState<string | null>(null);
  const showLogo = logo && failedLogo !== logo;

  return (
    <span aria-hidden="true" className={cn("chat-model-provider-logo", className)}>
      {showLogo ? (
        <Image
          src={logo}
          alt=""
          width={32}
          height={32}
          unoptimized
          className="h-full w-full object-contain"
          onError={() => setFailedLogo(logo)}
        />
      ) : (
        <span className="text-xs font-medium">{displayName.trim().charAt(0).toUpperCase() || "?"}</span>
      )}
    </span>
  );
}

interface ModelHubProvidersProps {
  providers: readonly string[];
  selectedProviders: readonly string[];
  onSelect: (provider: string | null) => void;
  loading?: boolean;
}

export default function ModelHubProviders({
  providers,
  selectedProviders,
  onSelect,
  loading = false,
}: ModelHubProvidersProps) {
  const availableProviders = Array.from(new Set(providers)).filter(Boolean);
  const selectedProvider = selectedProviders[0] ?? "";
  const initialLoading = loading && availableProviders.length === 0;
  const emptyLabel = initialLoading ? "Loading providers" : "All providers";
  const items = [
    { value: "", label: "All providers" },
    ...availableProviders.map((provider) => ({
      value: provider,
      label: getProviderLogoAndName(provider).displayName,
    })),
  ];

  return (
    <Select
      value={selectedProvider}
      items={items}
      onValueChange={(provider) => onSelect(provider || null)}
      disabled={initialLoading}
    >
      <SelectTrigger
        aria-label="Provider"
        aria-busy={loading}
        className="chat-model-filter-select chat-model-provider-select"
      >
        <SelectValue>
          {selectedProvider ? (
            <span className="flex min-w-0 items-center gap-2">
              <ProviderLogo provider={selectedProvider} className="chat-model-provider-logo-option" />
              <span className="truncate">{getProviderLogoAndName(selectedProvider).displayName}</span>
            </span>
          ) : (
            emptyLabel
          )}
        </SelectValue>
      </SelectTrigger>
      <SelectContent className="chat-theme" align="start">
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.value && <ProviderLogo provider={item.value} className="chat-model-provider-logo-option" />}
            <span>{item.label}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
