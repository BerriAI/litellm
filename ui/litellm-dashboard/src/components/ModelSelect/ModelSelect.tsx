import { ProxyModel, useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { Select, Skeleton, Tooltip, type SelectProps } from "antd";
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

const contextFilters: Record<ModelSelectProps["context"], (args: FilterContextArgs) => string[]> = {
  user: ({ allProxyModels, userModels, options }) => {
    if (!userModels) return [];
    if (options?.includeUserModels) return userModels;
    return [];
  },

  team: ({ allProxyModels, selectedOrganization, userModels }) => {
    if (selectedOrganization) {
      if (selectedOrganization.models.includes(MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value) || selectedOrganization.models.length === 0) {
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
  const { teamID, organizationID, options, context, dataTestId, value = [], onChange, style } = props;
  const { includeUserModels, showAllTeamModelsOption, showAllProxyModelsOverride, includeSpecialOptions } =
    options || {};
  const { data: allProxyModels, isLoading: isLoadingAllProxyModels } = useAllProxyModels();
  const { data: team, isLoading: isLoadingTeam } = useTeam(teamID);
  const { data: organization, isLoading: isLoadingOrganization } = useOrganization(organizationID);
  const { data: currentUser, isLoading: isCurrentUserLoading } = useCurrentUser();

  const isSpecialOption = (value: string) => MODEL_SELECT_SPECIAL_VALUES_ARRAY.some((sv) => sv.value === value);
  const hasSpecialOptionSelected = value.some(isSpecialOption);
  const isLoading = isLoadingAllProxyModels || isLoadingTeam || isLoadingOrganization || isCurrentUserLoading;
  const organizationHasAllProxyModels = organization?.models.includes(MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value) || organization?.models.length === 0;
  const shouldShowAllProxyModels =
    showAllProxyModelsOverride ||
    (organizationHasAllProxyModels && includeSpecialOptions) || context === "global";

  if (isLoading) {
    return <Skeleton.Input active block />;
  }

  const optionRender: NonNullable<SelectProps["optionRender"]> = (option) => {
    return <span>{option.label}</span>;
  };

  const handleChange = (values: string[]) => {
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
  return (
    <Select
      data-testid={dataTestId}
      value={value}
      onChange={handleChange}
      style={style}
      options={[
        includeSpecialOptions
          ? {
            label: <span>Special Options</span>,
            title: "Special Options",
            options: [
              ...(shouldShowAllProxyModels
                ? [
                  {
                    label: <span>All Proxy Models</span>,
                    value: MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
                    disabled:
                      value.length > 0 &&
                      value.some(
                        (v) => isSpecialOption(v) && v !== MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
                      ),
                    key: MODEL_SELECT_ALL_PROXY_MODELS_SPECIAL_VALUE.value,
                  },
                ]
                : []),
              {
                label: <span>No Default Models</span>,
                value: MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE.value,
                disabled:
                  value.length > 0 &&
                  value.some((v) => isSpecialOption(v) && v !== MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE.value),
                key: MODEL_SELECT_NO_DEFAULT_MODELS_SPECIAL_VALUE.value,
              },
            ],
          }
          : [],
        ...(wildcard.length > 0
          ? [
            {
              label: <span>Wildcard Options</span>,
              title: "Wildcard Options",
              options: wildcard.map((model) => {
                const provider = model.replace("/*", "");
                const capitalizedProvider = provider.charAt(0).toUpperCase() + provider.slice(1);

                return {
                  label: <span>{`All ${capitalizedProvider} models`}</span>,
                  value: model,
                  disabled: hasSpecialOptionSelected,
                };
              }),
            },
          ]
          : []),
        {
          label: <span>Models</span>,
          title: "Models",
          options: regular.map((model) => ({
            label: <span>{model}</span>,
            value: model,
            disabled: hasSpecialOptionSelected,
          })),
        },
      ]}
      mode="multiple"
      placeholder="Select Models"
      allowClear
      maxTagCount="responsive"
      maxTagPlaceholder={(omittedValues) => (
        <Tooltip
          styles={{ root: { pointerEvents: "none" } }}
          title={omittedValues.map(({ value }) => value).join(", ")}
        >
          <span>+{omittedValues.length} more</span>
        </Tooltip>
      )}
    />
  );
};
