import type { ModelDeprecationInfo } from "@/app/(dashboard)/hooks/models/useModelDeprecations";

export const deprecationBannerTitle = (
  deprecatedCount: number,
  imminentCount: number,
  warnWithinDays: number,
): string => {
  const deprecatedText =
    deprecatedCount === 0
      ? ""
      : `${deprecatedCount} ${deprecatedCount === 1 ? "model is" : "models are"} deprecated`;
  const imminentModelLabel = imminentCount === 1 ? "model" : "models";
  const imminentSubject =
    deprecatedCount > 0 ? `${imminentCount}` : `${imminentCount} ${imminentModelLabel}`;
  const imminentVerb = imminentCount === 1 ? "deprecates" : "deprecate";
  const imminentText =
    imminentCount === 0 ? "" : `${imminentSubject} ${imminentVerb} within ${warnWithinDays} days`;
  return [deprecatedText, imminentText].filter((text) => text.length > 0).join(", ");
};

export const deprecationEntryText = (info: ModelDeprecationInfo): string => {
  const model = `${info.model_name}${info.litellm_provider ? ` (${info.litellm_provider})` : ""}`;
  const successor = info.successor_model ? `, migrate to ${info.successor_model}` : "";

  if (info.status === "deprecated") {
    return `${model} was deprecated on ${info.deprecation_date}${successor}`;
  }

  const dayLabel = info.days_until_deprecation === 1 ? "day" : "days";
  return `${model} deprecates on ${info.deprecation_date}, in ${info.days_until_deprecation} ${dayLabel}${successor}`;
};
