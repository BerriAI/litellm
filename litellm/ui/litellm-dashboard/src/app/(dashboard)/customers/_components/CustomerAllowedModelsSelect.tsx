"use client";

import { useTranslation } from "react-i18next";
import { useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { providerForModel } from "@/app/(dashboard)/usage/_components/components/overview/modelProvider";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { customerModelOptions, type CustomerModelScope } from "./customerModelOptions";

interface Props {
  id: string;
  value: readonly string[];
  onChange: (value: string[]) => void;
  disabled?: boolean;
  modelScope?: CustomerModelScope;
  showHint?: boolean;
}

export default function CustomerAllowedModelsSelect({
  id,
  value,
  onChange,
  disabled,
  modelScope,
  showHint = true,
}: Props) {
  const { t } = useTranslation();
  const gatewayModels = useAllProxyModels({ enabled: !modelScope });
  const scopedModels = useQuery({
    queryKey: ["customer-model-options", modelScope],
    queryFn: () => customerModelOptions(modelScope!),
    enabled: Boolean(modelScope),
  });
  const models = modelScope ? scopedModels : gatewayModels;
  const configured = new Map((models.data?.data ?? []).map((model) => [model.id, model]));
  const options: MultiSelectOption[] = [...configured.values()].map((model) => ({
    value: model.id,
    label: model.id,
    icon: <ProviderLogo provider={providerForModel(model.id) ?? model.owned_by ?? ""} className="size-4" />,
  }));
  for (const selected of value) {
    if (configured.has(selected)) continue;
    const unavailableOption: MultiSelectOption = {
      value: selected,
      label: selected,
      disabled: true,
      description: models.data !== undefined ? t("customerPortal.hierarchy.unavailableModel") : undefined,
    };
    options.push(unavailableOption);
  }
  return (
    <div className="min-w-0 space-y-2 whitespace-normal">
      <MultiSelect
        id={id}
        options={options}
        value={[...value]}
        onValueChange={onChange}
        placeholder={t(
          modelScope ? "customerPortal.hierarchy.searchAppModels" : "customerPortal.hierarchy.searchModels",
        )}
        loading={models.isLoading}
        disabled={disabled}
      />
      {showHint && <p className="text-xs text-muted-foreground">{t("customerPortal.hierarchy.allowedModelsHint")}</p>}
      {models.isError && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground" role="status">
          <span>{t("customerPortal.hierarchy.modelsLoadError")}</span>
          <Button
            type="button"
            variant="link"
            size="sm"
            className="h-auto p-0 text-xs"
            onClick={() => void models.refetch()}
          >
            {t("customerPortal.hierarchy.retryModels")}
          </Button>
        </div>
      )}
    </div>
  );
}
