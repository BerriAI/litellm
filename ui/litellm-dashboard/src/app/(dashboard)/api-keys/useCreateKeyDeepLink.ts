import { parseAsString, parseAsStringLiteral, useQueryStates, type inferParserType } from "nuqs";
import { useCallback, useMemo } from "react";

import type { CreateKeyPrefillData } from "@/components/organisms/create_key_button";

const OWNED_BY_VALUES = ["you", "service_account", "another_user"] as const;
const KEY_TYPE_VALUES = ["default", "llm_api", "management"] as const;
const MAX_TEXT_LENGTH = 256;
const MAX_MODELS = 100;

const CREATE_KEY_DEEP_LINK_PARSERS = {
  create: parseAsString,
  owned_by: parseAsStringLiteral(OWNED_BY_VALUES),
  team_id: parseAsString,
  key_alias: parseAsString,
  models: parseAsString,
  key_type: parseAsStringLiteral(KEY_TYPE_VALUES),
};

type CreateKeyDeepLinkParams = inferParserType<typeof CREATE_KEY_DEEP_LINK_PARSERS>;

export interface CreateKeyDeepLink {
  autoOpenCreate: boolean;
  prefillData: CreateKeyPrefillData | undefined;
  clearDeepLink: () => void;
}

const parseModels = (models: string | null): string[] | undefined => {
  const parsed = (models ?? "")
    .split(",")
    .slice(0, MAX_MODELS)
    .map((model) => model.trim().slice(0, MAX_TEXT_LENGTH))
    .filter((model) => model.length > 0);
  return parsed.length > 0 ? parsed : undefined;
};

const toPrefillData = (params: CreateKeyDeepLinkParams): CreateKeyPrefillData | undefined => {
  const { owned_by, team_id, key_alias, models, key_type } = params;
  const hasPrefill = [owned_by, team_id, key_alias, models, key_type].some(Boolean);
  if (!hasPrefill) {
    return undefined;
  }
  return {
    owned_by: owned_by ?? undefined,
    team_id: team_id?.trim() || undefined,
    key_alias: key_alias ? key_alias.trim().slice(0, MAX_TEXT_LENGTH) : undefined,
    models: parseModels(models),
    key_type: key_type ?? undefined,
  };
};

export function useCreateKeyDeepLink(): CreateKeyDeepLink {
  const [params, setParams] = useQueryStates(CREATE_KEY_DEEP_LINK_PARSERS);
  const autoOpenCreate = params.create === "true";
  const prefillData = useMemo(() => (autoOpenCreate ? toPrefillData(params) : undefined), [autoOpenCreate, params]);
  const clearDeepLink = useCallback(() => void setParams(null), [setParams]);
  return { autoOpenCreate, prefillData, clearDeepLink };
}
