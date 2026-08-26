import { ProxyModel, useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxCollection,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxGroup,
  ComboboxItem,
  ComboboxLabel,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Organization, Team } from "../networking";
import { splitWildcardModels } from "./modelUtils";

const MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE = {
  label: "All Proxy Models",
  value: "all-proxy-models",
} as const;

const MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE = {
  label: "No Default Models",
  value: "no-default-models",
} as const;

export const MODEL_SENTINEL_OPTIONS = [
  MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE,
  MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE,
] as const;

const MAX_VISIBLE_MODEL_CHIPS = 5;

export interface ModelSelectProps {
  id?: string;
  teamID?: string;
  organizationID?: string;
  options?: {
    includeUserModels?: boolean;
    showAllTeamModelsOption?: boolean;
    showAllProxyModelsOverride?: boolean;
    includeSpecialOptions?: boolean;
  };
  context: "team" | "organization" | "user" | "global";
  dataTestId?: string;
  value?: string[];
  onChange: (values: string[]) => void;
  style?: React.CSSProperties;
}

type ModelOption = {
  label: string;
  value: string;
  disabled?: boolean;
};

type ModelOptionGroup = {
  label: string;
  items: ModelOption[];
};

type FilterContextArgs = {
  allProxyModels: string[];
  selectedTeam?: Team;
  selectedOrganization?: Organization;
  userModels?: string[];
  options?: ModelSelectProps["options"];
};

const contextFilters: Record<ModelSelectProps["context"], (args: FilterContextArgs) => string[]> = {
  user: ({ allProxyModels, userModels, options }) => {
    if (!userModels) return [];
    if (options?.includeUserModels) return userModels;
    return [];
  },

  team: ({ allProxyModels, selectedOrganization, userModels }) => {
    if (selectedOrganization) {
      if (
        selectedOrganization.models.includes(MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value) ||
        selectedOrganization.models.length === 0
      ) {
        return allProxyModels;
      }
      return allProxyModels.filter((model) => selectedOrganization.models.includes(model));
    }

    return allProxyModels ?? [];
  },

  organization: ({ allProxyModels }) => {
    return allProxyModels;
  },

  global: ({ allProxyModels }) => {
    return allProxyModels;
  },
};

const filterModels = (
  allProxyModels: ProxyModel[],
  ctx: ModelSelectProps,
  extra: { selectedTeam?: Team; selectedOrganization?: Organization; userModels?: string[] },
): string[] => {
  const deduplicatedProxyModels = Array.from(new Map(allProxyModels.map((m) => [m.id, m])).values()).map(
    (model) => model.id,
  );
  if (ctx.options?.showAllProxyModelsOverride) return deduplicatedProxyModels;

  const filterFn = contextFilters[ctx.context];
  if (!filterFn) return [];

  return filterFn({ allProxyModels: deduplicatedProxyModels, ...extra, options: ctx.options });
};

export const ModelSelect = (props: ModelSelectProps) => {
  const anchor = useComboboxAnchor();
  const { id, teamID, organizationID, options, context, dataTestId, value = [], onChange, style } = props;
  const { showAllProxyModelsOverride, includeSpecialOptions } = options || {};
  const { data: allProxyModels, isLoading: isLoadingAllProxyModels } = useAllProxyModels();
  const { data: team, isLoading: isLoadingTeam } = useTeam(teamID);
  const { data: organization, isLoading: isLoadingOrganization } = useOrganization(organizationID);
  const { data: currentUser, isLoading: isCurrentUserLoading } = useCurrentUser();

  const isSpecialOption = (value: string) => MODEL_SENTINEL_OPTIONS.some((sv) => sv.value === value);
  const hasSpecialOptionSelected = value.some(isSpecialOption);
  const isLoading = isLoadingAllProxyModels || isLoadingTeam || isLoadingOrganization || isCurrentUserLoading;
  const organizationHasAllProxyModels =
    organization?.models.includes(MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value) ||
    organization?.models.length === 0;
  const shouldShowAllProxyModels =
    showAllProxyModelsOverride || (organizationHasAllProxyModels && includeSpecialOptions) || context === "global";

  if (isLoading) {
    return <Skeleton className="h-9 w-full" />;
  }

  const handleChange = (selected: ModelOption[]) => {
    const values = selected.map((option) => option.value);
    const specialValues = values.filter(isSpecialOption);

    let finalValues: string[];
    if (specialValues.length > 0) {
      const lastSelectedSpecial = specialValues[specialValues.length - 1];
      finalValues = [lastSelectedSpecial];
    } else {
      finalValues = values;
    }

    onChange(finalValues);
  };

  const filteredModels = filterModels(allProxyModels?.data ?? [], props, {
    selectedTeam: team,
    selectedOrganization: organization,
    userModels: currentUser?.models,
  });

  const { wildcard, regular } = splitWildcardModels(filteredModels);

  const groups: ModelOptionGroup[] = [
    ...(includeSpecialOptions
      ? [
          {
            label: "Special Options",
            items: [
              ...(shouldShowAllProxyModels
                ? [
                    {
                      label: MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.label,
                      value: MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
                      disabled:
                        value.length > 0 &&
                        value.some(
                          (v) => isSpecialOption(v) && v !== MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
                        ),
                    },
                  ]
                : []),
              {
                label: MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE.label,
                value: MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE.value,
                disabled:
                  value.length > 0 &&
                  value.some((v) => isSpecialOption(v) && v !== MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE.value),
              },
            ],
          },
        ]
      : []),
    ...(wildcard.length > 0
      ? [
          {
            label: "Wildcard Options",
            items: wildcard.map((model) => {
              const provider = model.replace("/*", "");
              const capitalizedProvider = provider.charAt(0).toUpperCase() + provider.slice(1);

              return {
                label: `All ${capitalizedProvider} models`,
                value: model,
                disabled: hasSpecialOptionSelected,
              };
            }),
          },
        ]
      : []),
    {
      label: "Models",
      items: regular.map((model) => ({
        label: model,
        value: model,
        disabled: hasSpecialOptionSelected,
      })),
    },
  ];

  const optionsByValue = new Map(groups.flatMap((group) => group.items).map((option) => [option.value, option]));
  const selectedOptions = value.map((v) => optionsByValue.get(v) ?? { label: v, value: v });
  const overflowOptions = selectedOptions.slice(MAX_VISIBLE_MODEL_CHIPS);

  return (
    <TooltipProvider>
      <Combobox
        multiple
        items={groups}
        value={selectedOptions}
        onValueChange={handleChange}
        isItemEqualToValue={(option: ModelOption, selected: ModelOption) => option.value === selected.value}
        itemToStringLabel={(option: ModelOption) => option.label}
      >
        <ComboboxChips render={<div ref={anchor} />} data-testid={dataTestId} style={style} className="w-full">
          <ComboboxValue>
            {(selected: ModelOption[]) => (
              <>
                {selected.slice(0, MAX_VISIBLE_MODEL_CHIPS).map((option) => (
                  <ComboboxChip key={option.value} aria-label={option.label}>
                    {option.label}
                  </ComboboxChip>
                ))}
                {overflowOptions.length > 0 && (
                  <Tooltip>
                    <TooltipTrigger
                      render={<span className="px-1 text-xs text-muted-foreground" />}
                    >{`+${overflowOptions.length} more`}</TooltipTrigger>
                    <TooltipContent>{overflowOptions.map((option) => option.value).join(", ")}</TooltipContent>
                  </Tooltip>
                )}
              </>
            )}
          </ComboboxValue>
          <ComboboxChipsInput id={id} placeholder="Select Models" aria-label="Select Models" className="min-w-24" />
        </ComboboxChips>
        <ComboboxContent anchor={anchor}>
          <ComboboxEmpty>No models found</ComboboxEmpty>
          <ComboboxList>
            {(group: ModelOptionGroup) => (
              <ComboboxGroup key={group.label} items={group.items}>
                <ComboboxLabel>{group.label}</ComboboxLabel>
                <ComboboxCollection>
                  {(option: ModelOption) => (
                    <ComboboxItem key={option.value} value={option} disabled={option.disabled}>
                      <span className="min-w-0 break-words">{option.label}</span>
                    </ComboboxItem>
                  )}
                </ComboboxCollection>
              </ComboboxGroup>
            )}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
    </TooltipProvider>
  );
};
