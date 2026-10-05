import { useTracesLive } from "../api";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";

export type ProviderLookup = (model: string) => string | undefined;

const hasKnownLogo = (provider: string): boolean => getProviderLogoAndName(provider).logo !== "";

const costMapProvider = (costMap: unknown, model: string): string | undefined => {
  if (typeof costMap !== "object" || costMap === null) return undefined;
  const entry: unknown = Reflect.get(costMap, model);
  if (typeof entry !== "object" || entry === null) return undefined;
  const provider: unknown = Reflect.get(entry, "litellm_provider");
  return typeof provider === "string" && provider ? provider : undefined;
};

export const costMapLookup =
  (costMap: unknown): ProviderLookup =>
  (model) =>
    costMapProvider(costMap, model);

/** `openai/gpt-x` → "openai"; otherwise the cost map's provider for the full name, then the bare model. */
export function resolveSpanProvider(model: string | null, lookup: ProviderLookup): string | null {
  if (!model) return null;
  const slash = model.indexOf("/");
  const prefix = slash > 0 ? model.slice(0, slash) : "";
  if (prefix && hasKnownLogo(prefix)) return prefix;
  const bare = model.slice(model.lastIndexOf("/") + 1);
  return lookup(model) ?? lookup(bare) ?? null;
}

export function useSpanProvider(model: string | null): string | null {
  const { data } = useModelCostMap(useTracesLive() && model !== null);
  return resolveSpanProvider(model, costMapLookup(data));
}
