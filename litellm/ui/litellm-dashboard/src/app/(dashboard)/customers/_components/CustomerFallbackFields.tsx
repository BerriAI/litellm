import { useId } from "react";
import { ArrowDown, ArrowRight, ArrowUp, Plus, Trash2, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useAllProxyModels } from "@/app/(dashboard)/hooks/models/useModels";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import type { CustomerFallbackFields as CustomerFallbackRuleFields } from "./customerForms";

interface Props {
  fallbacks: CustomerFallbackRuleFields[] | null;
  onChange: (value: CustomerFallbackRuleFields[] | null) => void;
}

export default function CustomerFallbackFields({ fallbacks, onChange }: Props) {
  const { t } = useTranslation();
  const id = useId();
  const modelsQuery = useAllProxyModels();
  const modelGroups = [...new Set(modelsQuery.data?.data.map((model) => model.id) ?? [])].sort();
  const unselectedGroups = modelGroups.filter((model) => !fallbacks?.some((rule) => rule.model === model));
  const hasNoModels = !modelsQuery.isLoading && !modelsQuery.isError && modelGroups.length === 0;
  const updateRule = (index: number, changes: Partial<CustomerFallbackRuleFields>) =>
    onChange(fallbacks?.map((rule, ruleIndex) => (ruleIndex === index ? { ...rule, ...changes } : rule)) ?? null);
  const moveFallback = (ruleIndex: number, from: number, to: number) => {
    const targets = fallbacks?.[ruleIndex].fallbacks ?? [];
    updateRule(ruleIndex, {
      fallbacks: targets.map((model, index) => {
        if (index === from) return targets[to];
        if (index === to) return targets[from];
        return model;
      }),
    });
  };

  return (
    <section className="space-y-3" aria-label={t("customerPortal.fallbacks.title")}>
      <div className="flex items-center gap-2">
        <Switch
          id={`${id}-inherit`}
          checked={fallbacks === null}
          onCheckedChange={(checked) => onChange(checked ? null : [])}
        />
        <Label htmlFor={`${id}-inherit`}>{t("customerPortal.fallbacks.inherit")}</Label>
      </div>
      <p className="text-xs text-muted-foreground">
        {fallbacks === null ? t("customerPortal.fallbacks.inheritHelp") : t("customerPortal.fallbacks.customHelp")}
      </p>
      {fallbacks !== null && (
        <>
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
          {fallbacks.length === 0 && (
            <p className="text-sm text-muted-foreground">{t("customerPortal.fallbacks.empty")}</p>
          )}
          {fallbacks.map((rule, index) => {
            const primaryOptions = [...new Set([...unselectedGroups, rule.model])].map((model) => ({
              value: model,
              label: model,
            }));
            const unusedTargets = modelGroups.filter(
              (model) => model !== rule.model && !rule.fallbacks.includes(model),
            );
            const rowId = `${id}-${index}`;
            const changePrimary = (model: string | null) => {
              const isDuplicate = fallbacks.some((other, ruleIndex) => ruleIndex !== index && other.model === model);
              if (!model || isDuplicate) return;
              updateRule(index, { model, fallbacks: rule.fallbacks.filter((target) => target !== model) });
            };
            return (
              <div key={rule.model} className="space-y-3 py-1">
                <div className="grid items-start gap-3 sm:grid-cols-[minmax(0,1fr)_1.5rem_minmax(0,1.5fr)_2.5rem]">
                  <div className="min-w-0 space-y-2">
                    <Label htmlFor={`${rowId}-primary`} className="h-5 text-xs leading-5">
                      {t("customerPortal.fallbacks.primary")}
                    </Label>
                    <SearchSelect
                      inputId={`${rowId}-primary`}
                      aria-label={t("customerPortal.fallbacks.primaryForModel", { model: rule.model })}
                      options={primaryOptions}
                      value={rule.model}
                      allowClear={false}
                      onValueChange={changePrimary}
                      className="h-10"
                    />
                  </div>
                  <div
                    aria-hidden="true"
                    className="flex h-6 items-center justify-center text-muted-foreground sm:mt-7 sm:h-10"
                  >
                    <ArrowRight className="size-4 rotate-90 sm:rotate-0" />
                  </div>
                  <div className="min-w-0 space-y-3">
                    <ol
                      aria-label={t("customerPortal.fallbacks.backupsForModel", { model: rule.model })}
                      className="space-y-2"
                    >
                      {rule.fallbacks.map((model, targetIndex) => {
                        const targetId = `${rowId}-${targetIndex}`;
                        const targetOptions = [...new Set([...unusedTargets, model])].map((target) => ({
                          value: target,
                          label: target,
                        }));
                        const changeTarget = (target: string | null) => {
                          const isDuplicate = rule.fallbacks.some(
                            (other, otherIndex) => otherIndex !== targetIndex && other === target,
                          );
                          if (!target || target === rule.model || isDuplicate) return;
                          updateRule(index, {
                            fallbacks: rule.fallbacks.map((other, otherIndex) =>
                              otherIndex === targetIndex ? target : other,
                            ),
                          });
                        };
                        return (
                          <li key={model} className="space-y-2">
                            {targetIndex > 0 && (
                              <div aria-hidden="true" className="flex h-5 items-center pl-3 text-muted-foreground">
                                <ArrowDown className="size-4" />
                              </div>
                            )}
                            <Label htmlFor={targetId} className="h-5 text-xs leading-5">
                              {t("customerPortal.fallbacks.fallbackPosition", { position: targetIndex + 1 })}
                            </Label>
                            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-1">
                              <SearchSelect
                                inputId={targetId}
                                aria-label={t("customerPortal.fallbacks.targetForModel", {
                                  position: targetIndex + 1,
                                  model: rule.model,
                                })}
                                options={targetOptions}
                                value={model}
                                allowClear={false}
                                onValueChange={changeTarget}
                                className="h-10"
                              />
                              <div className="flex">
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="icon"
                                  className="h-10 w-10"
                                  aria-label={t("customerPortal.fallbacks.moveUp", { model })}
                                  disabled={targetIndex === 0}
                                  onClick={() => moveFallback(index, targetIndex, targetIndex - 1)}
                                >
                                  <ArrowUp className="size-4" />
                                </Button>
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="icon"
                                  className="h-10 w-10"
                                  aria-label={t("customerPortal.fallbacks.moveDown", { model })}
                                  disabled={targetIndex === rule.fallbacks.length - 1}
                                  onClick={() => moveFallback(index, targetIndex, targetIndex + 1)}
                                >
                                  <ArrowDown className="size-4" />
                                </Button>
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="icon"
                                  className="h-10 w-10"
                                  aria-label={t("customerPortal.fallbacks.removeTarget", { model })}
                                  onClick={() =>
                                    updateRule(index, {
                                      fallbacks: rule.fallbacks.filter((_, otherIndex) => otherIndex !== targetIndex),
                                    })
                                  }
                                >
                                  <X className="size-4" />
                                </Button>
                              </div>
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                    {rule.fallbacks.length === 0 && (
                      <p className="text-xs text-muted-foreground">{t("customerPortal.fallbacks.selectBackup")}</p>
                    )}
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={modelsQuery.isLoading || modelsQuery.isError || unusedTargets.length === 0}
                      onClick={() => updateRule(index, { fallbacks: [...rule.fallbacks, unusedTargets[0]] })}
                    >
                      <Plus className="size-4" />
                      {t("customerPortal.fallbacks.addTarget")}
                    </Button>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-10 w-10 justify-self-end sm:mt-7"
                    aria-label={t("customerPortal.fallbacks.remove", { model: rule.model })}
                    onClick={() => onChange(fallbacks.filter((_, ruleIndex) => ruleIndex !== index))}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </div>
              </div>
            );
          })}
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={modelsQuery.isLoading || modelsQuery.isError || unselectedGroups.length === 0}
            onClick={() => onChange([...fallbacks, { model: unselectedGroups[0], fallbacks: [] }])}
          >
            <Plus className="size-4" /> {t("customerPortal.fallbacks.add")}
          </Button>
        </>
      )}
    </section>
  );
}
