import { ProxyModel, useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { useTeam } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Select, Skeleton, Tooltip, type SelectProps } from "antd";
import { Organization, Team } from "../networking";
import { useOrganization } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { splitWildcardModels } from "./modelUtils";

const MODEL_SELECT_SPECIAL_VALUES = {
  ALL_PROXY_MODELS: {
    label: "All Proxy Models",
    value: "all-proxy-models",
  },
  NO_DEFAULT_MODELS: {
    label: "No Default Models",
    value: "no-default-models",
  },
};

const MODEL_SELECT_SPECIAL_VALUES_ARRAY = Object.values(MODEL_SELECT_SPECIAL_VALUES);

export interface ModelSelectContext {
  teamID?: string;
  organizationID?: string;
  includeUserModels?: boolean;
  showAllTeamModelsOption?: boolean;
  showAllProxyModelsOverride?: boolean;
  includeSpecialOptions?: boolean;
  dataTestId?: string;
  value?: string[];
  onChange: (values: string[]) => void;
}

const filterModels = (
  allProxyModels: ProxyModel[],
  ctx: ModelSelectContext,
  {
    selectedTeam,
    selectedOrganization,
    userModels,
  }: { selectedTeam?: Team; selectedOrganization?: Organization; userModels?: ProxyModel[] },
): ProxyModel[] => {
  const deduplicatedProxyModels = Array.from(new Map(allProxyModels.map((model) => [model.id, model])).values());
  if (ctx.showAllProxyModelsOverride) {
    return deduplicatedProxyModels;
  }

  if (selectedOrganization) {
    if (selectedOrganization.models.includes(MODEL_SELECT_SPECIAL_VALUES.ALL_PROXY_MODELS.value)) {
      return deduplicatedProxyModels;
    }
  }

  return [];
};

export const ModelSelect = (ctx: ModelSelectContext) => {
  const {
    teamID,
    organizationID,
    includeUserModels,
    showAllTeamModelsOption,
    showAllProxyModelsOverride,
    includeSpecialOptions,
    dataTestId,
    value = [],
    onChange,
  } = ctx;
  const { data: allProxyModels, isLoading: isLoadingAllProxyModels } = useAllProxyModels();
  const { data: team, isLoading: isLoadingTeam } = useTeam(teamID);
  const { data: organization, isLoading: isLoadingOrganization } = useOrganization(organizationID);

  const isSpecialOption = (value: string) => MODEL_SELECT_SPECIAL_VALUES_ARRAY.some((sv) => sv.value === value);
  const hasSpecialOptionSelected = value.some(isSpecialOption);
  const isLoading = isLoadingAllProxyModels || isLoadingTeam || isLoadingOrganization;

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

  const filteredModels = filterModels(allProxyModels?.data ?? [], ctx, {
    selectedTeam: team,
    selectedOrganization: organization,
  });

  const { wildcard, regular } = splitWildcardModels(filteredModels);
  return (
    <Select
      data-testid={dataTestId}
      value={value}
      onChange={handleChange}
      options={[
        {
          label: <span>Special Options</span>,
          title: "Special Options",
          options: MODEL_SELECT_SPECIAL_VALUES_ARRAY.map((specialValue) => ({
            label: <span>{specialValue.label}</span>,
            value: specialValue.value,
            disabled: value.length > 0 && value.some((v) => isSpecialOption(v) && v !== specialValue.value),
            key: specialValue.value,
          })),
        },
        {
          label: <span>Wildcard Options</span>,
          title: "Wildcard Options",
          options: wildcard.map((model) => {
            const provider = model.id.replace("/*", "");
            const capitalizedProvider = provider.charAt(0).toUpperCase() + provider.slice(1);

            return {
              label: <span>{`All ${capitalizedProvider} models`}</span>,
              value: model.id,
              disabled: hasSpecialOptionSelected,
            };
          }),
        },
        {
          label: <span>Models</span>,
          title: "Models",
          options: regular.map((model) => ({
            label: <span>{model.id}</span>,
            value: model.id,
            disabled: hasSpecialOptionSelected,
          })),
        },
      ]}
      mode="multiple"
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
