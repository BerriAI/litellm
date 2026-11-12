import { Plus, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { ModelRateLimitFields } from "./customerForms";
import RateLimitInput from "./RateLimitInput";

interface Props {
  idPrefix: string;
  tpm: string;
  rpm: string;
  onTpmChange: (value: string) => void;
  onRpmChange: (value: string) => void;
  modelLimits: ModelRateLimitFields[];
  onModelLimitsChange: (value: ModelRateLimitFields[]) => void;
}

export default function CustomerRateLimitFields({
  idPrefix,
  tpm,
  rpm,
  onTpmChange,
  onRpmChange,
  modelLimits,
  onModelLimitsChange,
}: Props) {
  const { t } = useTranslation();
  const modelsQuery = useAllProxyModels();
  const modelGroups = [...new Set(modelsQuery.data?.data.map((model) => model.id) ?? [])].sort();
  const unselectedGroups = modelGroups.filter((model) => !modelLimits.some((limit) => limit.model === model));
  const hasNoModels = !modelsQuery.isLoading && !modelsQuery.isError && modelGroups.length === 0;
  const updateRow = (index: number, changes: Partial<ModelRateLimitFields>) =>
    onModelLimitsChange(modelLimits.map((limit, rowIndex) => (rowIndex === index ? { ...limit, ...changes } : limit)));

  return (
    <div className="space-y-6">
      <section className="space-y-3" aria-labelledby={`${idPrefix}-overall-title`}>
        <h3 id={`${idPrefix}-overall-title`} className="text-xs font-medium text-muted-foreground">
          {t("customerPortal.rateLimits.overallTitle")}
        </h3>
        <div className="grid gap-3 sm:grid-cols-2">
          <RateLimitInput
            id={`${idPrefix}-tpm`}
            label={t("customerPortal.rateLimits.tpm")}
            value={tpm}
            onChange={onTpmChange}
          />
          <RateLimitInput
            id={`${idPrefix}-rpm`}
            label={t("customerPortal.rateLimits.rpm")}
            value={rpm}
            onChange={onRpmChange}
          />
        </div>
        <p className="text-xs text-muted-foreground">{t("customerPortal.rateLimits.numberHelp")}</p>
      </section>
      <section className="space-y-3" aria-labelledby={`${idPrefix}-model-title`}>
        <h3 id={`${idPrefix}-model-title`} className="text-xs font-medium text-muted-foreground">
          {t("customerPortal.rateLimits.modelTitle")}
        </h3>
        {modelsQuery.isLoading && (
          <p role="status" className="text-sm text-muted-foreground">
            {t("customerPortal.rateLimits.loading")}
          </p>
        )}
        {modelsQuery.isError && (
          <div role="alert" className="flex items-center gap-2 text-sm text-destructive">
            {t("customerPortal.rateLimits.loadError")}
            <Button type="button" variant="link" size="sm" onClick={() => void modelsQuery.refetch()}>
              {t("customerPortal.rateLimits.retry")}
            </Button>
          </div>
        )}
        {hasNoModels && <p className="text-sm text-muted-foreground">{t("customerPortal.rateLimits.noModels")}</p>}
        {modelLimits.map((limit, index) => {
          const options = [...new Set([...unselectedGroups, limit.model])].map((model) => ({
            value: model,
            label: model,
          }));
          const rowPrefix = `${idPrefix}-${index}`;
          return (
            <div
              key={limit.model}
              className="grid items-start gap-3 sm:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)_minmax(0,1fr)_2.5rem]"
            >
              <div className="min-w-0 space-y-2">
                <Label htmlFor={`${rowPrefix}-group`} className="h-5 text-xs leading-5">
                  {t("customerPortal.rateLimits.modelGroup")}
                </Label>
                <Select
                  items={options}
                  value={limit.model}
                  onValueChange={(model) => {
                    if (model && !modelLimits.some((other, rowIndex) => rowIndex !== index && other.model === model)) {
                      updateRow(index, { model });
                    }
                  }}
                >
                  <SelectTrigger
                    id={`${rowPrefix}-group`}
                    aria-label={t("customerPortal.rateLimits.groupForModel", { model: limit.model })}
                    className="h-10 w-full min-w-0 data-[size=default]:h-10"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {options.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <RateLimitInput
                id={`${rowPrefix}-tpm`}
                label={t("customerPortal.rateLimits.tpm")}
                ariaLabel={t("customerPortal.rateLimits.tpmForModel", { model: limit.model })}
                value={limit.tpm}
                onChange={(value) => updateRow(index, { tpm: value })}
              />
              <RateLimitInput
                id={`${rowPrefix}-rpm`}
                label={t("customerPortal.rateLimits.rpm")}
                ariaLabel={t("customerPortal.rateLimits.rpmForModel", { model: limit.model })}
                value={limit.rpm}
                onChange={(value) => updateRow(index, { rpm: value })}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-10 w-10 justify-self-end sm:mt-7"
                aria-label={t("customerPortal.rateLimits.removeModel", { model: limit.model })}
                onClick={() => onModelLimitsChange(modelLimits.filter((_, rowIndex) => rowIndex !== index))}
              >
                <X className="size-4" />
              </Button>
            </div>
          );
        })}
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={modelsQuery.isLoading || modelsQuery.isError || unselectedGroups.length === 0}
          onClick={() => onModelLimitsChange([...modelLimits, { model: unselectedGroups[0], tpm: "", rpm: "" }])}
        >
          <Plus className="size-4" /> {t("customerPortal.rateLimits.addModel")}
        </Button>
      </section>
    </div>
  );
}
