import { tagListCall } from "@/components/networking";
import type { EntityType } from "@/components/EntityUsageExport/types";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import { useEffect, useState } from "react";

interface ScopedTagOptionsInput {
  entityType: EntityType;
  selectedTeamIds: readonly string[];
  selectedTags: readonly string[];
  hasRequestWindow: boolean;
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
}

export function useScopedTagOptions({
  entityType,
  selectedTeamIds,
  selectedTags,
  hasRequestWindow,
  accessToken,
  startTime,
  endTime,
}: ScopedTagOptionsInput): SearchSelectOption[] | null {
  const [options, setOptions] = useState<SearchSelectOption[] | null>(null);

  useEffect(() => {
    const scopeTeams = entityType === "team" ? selectedTags : selectedTeamIds;
    const optionsNeeded = entityType === "team" || (entityType === "tag" && scopeTeams.length > 0);
    if (!optionsNeeded || !hasRequestWindow) {
      return;
    }
    let cancelled = false;
    tagListCall(accessToken as string, startTime, endTime, {
      teamIds: scopeTeams.length > 0 ? scopeTeams : undefined,
      usageOnly: true,
    })
      .then((tags) => {
        if (!cancelled) {
          // /tag/list returns an array of tags, so read names from the values, not the keys (indices).
          setOptions(Object.values(tags).map((tag) => ({ label: tag.name, value: tag.name })));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setOptions([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [entityType, selectedTeamIds, selectedTags, hasRequestWindow, accessToken, startTime, endTime]);

  return options;
}
