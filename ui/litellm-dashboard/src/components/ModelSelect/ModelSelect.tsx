import React, { useMemo, useState } from "react";
import { ProxyModel, useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Check, X } from "lucide-react";
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

const MODEL_SELECT_SPECIAL_VALUES_ARRAY = [
  MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE,
  MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE,
] as const;

export interface ModelSelectProps {
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

type FilterContextArgs = {
  allProxyModels: string[];
  selectedTeam?: Team;
  selectedOrganization?: Organization;
  userModels?: string[];
  options?: ModelSelectProps["options"];
};

const contextFilters: Record<
  ModelSelectProps["context"],
  (args: FilterContextArgs) => string[]
> = {
  user: ({ userModels, options }) => {
    if (!userModels) return [];
    if (options?.includeUserModels) return userModels;
    return [];
  },

  team: ({ allProxyModels, selectedOrganization }) => {
    if (selectedOrganization) {
      if (
        selectedOrganization.models.includes(
          MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
        ) ||
        selectedOrganization.models.length === 0
      ) {
        return allProxyModels;
      }
      return allProxyModels.filter((model) =>
        selectedOrganization.models.includes(model),
      );
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
  extra: {
    selectedTeam?: Team;
    selectedOrganization?: Organization;
    userModels?: string[];
  },
): string[] => {
  const deduplicatedProxyModels = Array.from(
    new Map(allProxyModels.map((m) => [m.id, m])).values(),
  ).map((model) => model.id);
  if (ctx.options?.showAllProxyModelsOverride) return deduplicatedProxyModels;
  const filterFn = contextFilters[ctx.context];
  if (!filterFn) return [];
  return filterFn({
    allProxyModels: deduplicatedProxyModels,
    ...extra,
    options: ctx.options,
  });
};

interface OptionEntry {
  label: React.ReactNode;
  value: string;
  disabled?: boolean;
}

interface OptionGroup {
  label: string;
  options: OptionEntry[];
}

export const ModelSelect = (props: ModelSelectProps) => {
  const {
    teamID,
    organizationID,
    options,
    context,
    dataTestId,
    value = [],
    onChange,
    style,
  } = props;
  const { includeSpecialOptions, showAllProxyModelsOverride } = options || {};

  const { data: allProxyModels, isLoading: isLoadingAllProxyModels } =
    useAllProxyModels();
  const { data: team, isLoading: isLoadingTeam } = useTeam(teamID);
  const { data: organization, isLoading: isLoadingOrganization } =
    useOrganization(organizationID);
  const { data: currentUser, isLoading: isCurrentUserLoading } =
    useCurrentUser();

  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  const isSpecialOption = (v: string) =>
    MODEL_SELECT_SPECIAL_VALUES_ARRAY.some((sv) => sv.value === v);
  const hasSpecialOptionSelected = value.some(isSpecialOption);
  const isLoading =
    isLoadingAllProxyModels ||
    isLoadingTeam ||
    isLoadingOrganization ||
    isCurrentUserLoading;
  const organizationHasAllProxyModels =
    organization?.models.includes(
      MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
    ) || organization?.models.length === 0;
  const shouldShowAllProxyModels =
    showAllProxyModelsOverride ||
    (organizationHasAllProxyModels && includeSpecialOptions) ||
    context === "global";

  const filteredModels = filterModels(allProxyModels?.data ?? [], props, {
    selectedTeam: team,
    selectedOrganization: organization,
    userModels: currentUser?.models,
  });

  const { wildcard, regular } = splitWildcardModels(filteredModels);

  const optionGroups: OptionGroup[] = useMemo(() => {
    const groups: OptionGroup[] = [];

    if (includeSpecialOptions) {
      const specialEntries: OptionEntry[] = [];
      if (shouldShowAllProxyModels) {
        specialEntries.push({
          label: "All Proxy Models",
          value: MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
          disabled:
            value.length > 0 &&
            value.some(
              (v) =>
                isSpecialOption(v) &&
                v !== MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
            ),
        });
      }
      specialEntries.push({
        label: "No Default Models",
        value: MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE.value,
        disabled:
          value.length > 0 &&
          value.some(
            (v) =>
              isSpecialOption(v) &&
              v !== MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE.value,
          ),
      });
      groups.push({ label: "Special Options", options: specialEntries });
    }

    if (wildcard.length > 0) {
      groups.push({
        label: "Wildcard Options",
        options: wildcard.map((model) => {
          const provider = model.replace("/*", "");
          const cap = provider.charAt(0).toUpperCase() + provider.slice(1);
          return {
            label: `All ${cap} models`,
            value: model,
            disabled: hasSpecialOptionSelected,
          };
        }),
      });
    }

    groups.push({
      label: "Models",
      options: regular.map((model) => ({
        label: model,
        value: model,
        disabled: hasSpecialOptionSelected,
      })),
    });

    return groups;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    includeSpecialOptions,
    shouldShowAllProxyModels,
    wildcard.join(","),
    regular.join(","),
    value.join(","),
    hasSpecialOptionSelected,
  ]);

  if (isLoading) {
    return <Skeleton className="h-10 w-full" />;
  }

  const selectOption = (val: string) => {
    let finalValues: string[];
    if (isSpecialOption(val)) {
      // Selecting a special option replaces the full selection.
      finalValues = value.includes(val) ? value.filter((v) => v !== val) : [val];
    } else if (value.includes(val)) {
      finalValues = value.filter((v) => v !== val);
    } else {
      // Adding a normal model — strip any special options first.
      finalValues = [...value.filter((v) => !isSpecialOption(v)), val];
    }
    onChange(finalValues);
  };

  const valueToLabel = (v: string): string => {
    for (const g of optionGroups) {
      const found = g.options.find((o) => o.value === v);
      if (found)
        return typeof found.label === "string" ? found.label : v;
    }
    return v;
  };

  const displayLabels = value.map(valueToLabel);
  const maxVisibleChips = 3;
  const visibleChips = displayLabels.slice(0, maxVisibleChips);
  const hiddenChips = displayLabels.slice(maxVisibleChips);

  const matchesSearch = (label: string): boolean => {
    if (!search) return true;
    return label.toLowerCase().includes(search.toLowerCase());
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          data-testid={dataTestId}
          style={style}
          className={cn(
            "min-h-10 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left focus:outline-none focus:ring-2 focus:ring-ring",
          )}
        >
          {value.length === 0 ? (
            <span className="text-muted-foreground px-1">Select Models</span>
          ) : (
            <>
              {visibleChips.map((label, idx) => (
                <Badge
                  key={value[idx]}
                  variant="secondary"
                  className="gap-1 inline-flex items-center"
                >
                  {label}
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation();
                      selectOption(value[idx]);
                    }}
                    className="inline-flex items-center"
                    aria-label={`Remove ${label}`}
                  >
                    <X size={12} />
                  </span>
                </Badge>
              ))}
              {hiddenChips.length > 0 && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="text-xs text-muted-foreground px-1">
                        +{hiddenChips.length} more
                      </span>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-sm">
                      {hiddenChips.join(", ")}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </>
          )}
          {value.length > 0 && (
            <span
              role="button"
              tabIndex={0}
              aria-label="Clear all"
              onClick={(e) => {
                e.stopPropagation();
                onChange([]);
              }}
              className="ml-auto text-muted-foreground inline-flex items-center"
            >
              <X size={14} />
            </span>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-[--radix-popover-trigger-width] max-w-none">
        <div className="flex items-center border-b border-border p-2">
          <Input
            value={search}
            placeholder="Search models..."
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 border-0 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
          />
        </div>
        <div className="max-h-60 overflow-y-auto p-1">
          {optionGroups.map((group) => {
            const items = group.options.filter((o) =>
              typeof o.label === "string" ? matchesSearch(o.label) : true,
            );
            if (items.length === 0) return null;
            return (
              <div key={group.label} className="mb-2">
                <div className="text-[11px] text-muted-foreground px-2 py-1 font-semibold uppercase tracking-wide">
                  {group.label}
                </div>
                {items.map((o) => {
                  const selected = value.includes(o.value);
                  return (
                    <button
                      key={o.value}
                      type="button"
                      disabled={o.disabled}
                      onClick={() => selectOption(o.value)}
                      className={cn(
                        "flex items-center gap-2 w-full text-left text-sm px-2 py-1.5 rounded-sm hover:bg-muted",
                        o.disabled && "opacity-50 cursor-not-allowed hover:bg-transparent",
                      )}
                    >
                      <span
                        className={cn(
                          "h-4 w-4 shrink-0 inline-flex items-center justify-center rounded-sm border border-primary",
                          selected && "bg-primary text-primary-foreground",
                        )}
                      >
                        {selected && <Check className="h-3 w-3" />}
                      </span>
                      <span className="truncate">{o.label}</span>
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      </PopoverContent>
    </Popover>
  );
};
