"use client";

import { LayoutGrid } from "lucide-react";
import Image from "next/image";
import { useState } from "react";

import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
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

  return (
    <section className="space-y-3">
      <h2 className="text-sm font-medium text-foreground">Available providers</h2>
      <div className="chat-model-providers" role="group" aria-label="Available providers" aria-busy={loading}>
        <Button
          type="button"
          variant="ghost"
          className="chat-model-provider-button"
          aria-pressed={selectedProviders.length === 0}
          onClick={() => onSelect(null)}
        >
          <span className="chat-model-provider-logo" aria-hidden="true">
            <LayoutGrid className="size-4" />
          </span>
          <span>All models</span>
        </Button>
        {availableProviders.map((provider) => (
          <Button
            key={provider}
            type="button"
            variant="ghost"
            className="chat-model-provider-button"
            aria-pressed={selectedProviders.includes(provider)}
            onClick={() => onSelect(provider)}
          >
            <ProviderLogo provider={provider} />
            <span>{getProviderLogoAndName(provider).displayName}</span>
          </Button>
        ))}
        {loading && availableProviders.length === 0 && (
          <>
            <span role="status" className="sr-only">
              Loading providers
            </span>
            {[0, 1, 2].map((index) => (
              <Skeleton key={index} className="h-16 w-36 rounded-xl" aria-hidden="true" />
            ))}
          </>
        )}
      </div>
    </section>
  );
}
