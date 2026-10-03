import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { cn } from "@/lib/cva.config";

import { modelName, providerOf } from "../../model/live";

const MARK_CROP: Readonly<Record<string, string>> = {
  cerebras: "absolute top-[-8px] left-[-2.4px] h-[30px] w-[40px] max-w-none",
};

export function ProviderLogo({ model, className }: { model: string; className?: string }) {
  const provider = providerOf(model);
  const { logo, displayName } = getProviderLogoAndName(provider);
  if (!provider || !logo) return null;
  const crop = MARK_CROP[provider];
  return (
    <span className={cn("relative inline-block size-3.5 shrink-0 overflow-hidden", className)}>
      {/* eslint-disable-next-line @next/next/no-img-element -- provider logos are static svgs from /assets/logos */}
      <img src={logo} alt={displayName} className={crop ?? "size-full object-contain"} />
    </span>
  );
}

export function ProviderBadge({ model }: { model: string }) {
  if (!model) return null;
  const provider = providerOf(model);
  const { displayName } = getProviderLogoAndName(provider);
  return (
    <span
      data-testid="provider-badge"
      className="inline-flex min-w-0 items-center gap-1.5 rounded-full border border-(--provider)/30 bg-(--provider)/5 px-2.5 py-1 text-xs font-medium text-(--provider)"
    >
      <ProviderLogo model={model} />
      <span className="truncate">
        {provider ? `Analyzed by ${displayName} · ${modelName(model)}` : `Analyzed by ${model}`}
      </span>
    </span>
  );
}
