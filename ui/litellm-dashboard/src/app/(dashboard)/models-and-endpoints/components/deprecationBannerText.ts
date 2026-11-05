import type { ModelDeprecationInfo } from "@/app/(dashboard)/hooks/models/useModelDeprecations";

const shownModelName = (info: ModelDeprecationInfo): string => info.team_public_model_name ?? info.model_name;

export const countDistinctModels = (infos: readonly ModelDeprecationInfo[]): number =>
  new Set(infos.map((info) => JSON.stringify([shownModelName(info), info.litellm_provider ?? null]))).size;

export const parseLocalDate = (value: string): Date => {
  const [year, month, day] = value.split("-");
  return new Date(Number(year), Number(month) - 1, Number(day));
};

const startOfLocalDay = (today: Date): Date => new Date(today.getFullYear(), today.getMonth(), today.getDate());

export const localDaysUntil = (deprecationDate: string, today: Date): number =>
  Math.round((parseLocalDate(deprecationDate).getTime() - startOfLocalDay(today).getTime()) / 86_400_000);

export const bucketByLocalDate = (
  infos: readonly ModelDeprecationInfo[],
  today: Date,
  warnWithinDays: number,
): { deprecated: ModelDeprecationInfo[]; imminent: ModelDeprecationInfo[] } => {
  const inWindow = infos
    .map((info): ModelDeprecationInfo => {
      const daysUntilDeprecation = localDaysUntil(info.deprecation_date, today);
      return {
        ...info,
        days_until_deprecation: daysUntilDeprecation,
        status: daysUntilDeprecation < 0 ? "deprecated" : "imminent",
      };
    })
    .filter((info) => info.days_until_deprecation <= warnWithinDays);

  return {
    deprecated: inWindow.filter((info) => info.status === "deprecated"),
    imminent: inWindow.filter((info) => info.status === "imminent"),
  };
};

const hasSameDedupeKey = (candidate: ModelDeprecationInfo, info: ModelDeprecationInfo): boolean => {
  const sameDisplayedModel = shownModelName(candidate) === shownModelName(info);
  const sameProvider = (candidate.litellm_provider ?? null) === (info.litellm_provider ?? null);
  const sameDeprecationDate = candidate.deprecation_date === info.deprecation_date;
  const sameSuccessor = (candidate.successor_model ?? null) === (info.successor_model ?? null);
  const sameVisibleIdentity = sameDisplayedModel && sameProvider;
  const sameDeprecationIdentity = sameDeprecationDate && sameSuccessor;
  return sameVisibleIdentity && sameDeprecationIdentity;
};

export const dedupeDeprecationEntries = (infos: readonly ModelDeprecationInfo[]): ModelDeprecationInfo[] =>
  infos.filter((info, index) => infos.findIndex((candidate) => hasSameDedupeKey(candidate, info)) === index);

export const deprecationBannerFingerprint = (infos: readonly ModelDeprecationInfo[]): string =>
  JSON.stringify(
    infos
      .map((info) =>
        JSON.stringify([
          shownModelName(info),
          info.litellm_provider ?? null,
          info.deprecation_date,
          info.status,
          info.successor_model ?? null,
        ]),
      )
      .sort(),
  );

export const deprecationBannerTitle = (
  deprecatedCount: number,
  imminentCount: number,
  warnWithinDays: number,
): string => {
  const deprecatedText =
    deprecatedCount === 0 ? "" : `${deprecatedCount} ${deprecatedCount === 1 ? "model is" : "models are"} deprecated`;
  const imminentModelLabel = imminentCount === 1 ? "model" : "models";
  const imminentSubject = deprecatedCount > 0 ? `${imminentCount}` : `${imminentCount} ${imminentModelLabel}`;
  const imminentVerb = imminentCount === 1 ? "deprecates" : "deprecate";
  const imminentText = imminentCount === 0 ? "" : `${imminentSubject} ${imminentVerb} within ${warnWithinDays} days`;
  return [deprecatedText, imminentText].filter((text) => text.length > 0).join(", ");
};

export const deprecationEntryText = (info: ModelDeprecationInfo): string => {
  const model = `${shownModelName(info)}${info.litellm_provider ? ` (${info.litellm_provider})` : ""}`;
  const successor = info.successor_model ? `, migrate to ${info.successor_model}` : "";

  if (info.status === "deprecated") {
    return `${model} was deprecated on ${info.deprecation_date}${successor}`;
  }

  if (info.days_until_deprecation === 0) {
    return `${model} deprecates today (${info.deprecation_date})${successor}`;
  }

  const dayLabel = info.days_until_deprecation === 1 ? "day" : "days";
  return `${model} deprecates on ${info.deprecation_date}, in ${info.days_until_deprecation} ${dayLabel}${successor}`;
};
