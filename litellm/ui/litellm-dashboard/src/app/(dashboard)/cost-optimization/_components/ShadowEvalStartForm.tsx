"use client";

import React, { useMemo, useState } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { useWatch } from "react-hook-form";
import { useZodForm } from "@/lib/forms/useZodForm";

import { useInfiniteKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  useAutoRouters,
  usePlainChatModelDeployments,
  usePlainChatModelGroups,
  usePlainModelGroups,
} from "@/app/(dashboard)/hooks/models/useModels";
import { buildModelAvailability, deploymentRefsFromModelInfo, resolveAvailableModels } from "@/lib/autorouter_presets";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { PaginatedMultiSelect } from "@/components/shared/PaginatedMultiSelect";
import TeamMultiSelect from "@/components/common_components/team_multi_select";
import { userOptionLabel } from "@/components/common_components/UserDropdown";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import { useStartShadowEval } from "./useShadowEval";
import { buildStartBody } from "./shadow-eval/mapper";
import {
  MAX_MODELS,
  MAX_ROUTERS,
  shadowEvalDefaults,
  shadowEvalSchema,
  parseStartNumbers,
  type ShadowEvalDirection,
} from "./shadow-eval/schema";
const RECOMMENDED_JUDGE_MODELS = ["anthropic/claude-sonnet-5", "openai/gpt-4o", "gemini/gemini-2.5-pro"] as const;

const DIRECTION_OPTIONS: readonly { value: ShadowEvalDirection; labelKey: ParseKeys }[] = [
  { value: "forward", labelKey: "costOptimization.shadowEval.start.directionOptions.forward" },
  { value: "reverse", labelKey: "costOptimization.shadowEval.start.directionOptions.reverse" },
] as const;

const START_FORM_DESCRIPTION: Record<ShadowEvalDirection, ParseKeys> = {
  forward: "costOptimization.shadowEval.start.description.forward",
  reverse: "costOptimization.shadowEval.start.description.reverse",
};

const DURATION_OPTIONS = [
  { value: "1", days: 1 },
  { value: "3", days: 3 },
  { value: "7", days: 7 },
  { value: "14", days: 14 },
  { value: "30", days: 30 },
] as const;

const Field: React.FC<{ label: string; htmlFor?: string; className?: string; children: React.ReactNode }> = ({
  label,
  htmlFor,
  className,
  children,
}) => (
  <div className={`space-y-1.5 ${className ?? ""}`}>
    <Label htmlFor={htmlFor} className="text-xs">
      {label}
    </Label>
    {children}
  </div>
);

const KeySelect: React.FC<{ value: string[]; onChange: (tokens: string[]) => void }> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const [search, setSearch] = useState("");
  const { data, isPending, isError, fetchNextPage, hasNextPage, isFetchingNextPage } = useInfiniteKeys(50, {
    selectedKeyAlias: search || null,
  });
  const options = useMemo<SearchSelectOption[]>(
    () =>
      (data?.pages ?? [])
        .flatMap((page) => page.keys)
        .map((key) => ({
          label: key.key_alias || key.key_name || key.token,
          value: key.token,
          sublabel: key.token,
        })),
    [data],
  );
  return (
    <PaginatedMultiSelect
      inputId="shadow-eval-key"
      options={options}
      value={value}
      onValueChange={onChange}
      onSearchChange={setSearch}
      onLoadMore={() => void fetchNextPage()}
      hasNextPage={hasNextPage}
      isFetchingNextPage={isFetchingNextPage}
      isLoading={isPending}
      placeholder={t("costOptimization.shadowEval.start.keys.placeholder")}
      emptyText={t("costOptimization.shadowEval.start.keys.empty")}
      errorText={isError ? t("costOptimization.shadowEval.start.keys.loadFailed") : undefined}
    />
  );
};

const UserSelect: React.FC<{ value: string[]; onChange: (ids: string[]) => void }> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const [search, setSearch] = useState("");
  const { data, isPending, isError, fetchNextPage, hasNextPage, isFetchingNextPage } = useInfiniteUsers(
    50,
    search || undefined,
  );
  const options = useMemo<SearchSelectOption[]>(
    () =>
      Array.from(
        new Map(
          (data?.pages ?? [])
            .flatMap((page) => page.users)
            .map((user) => [user.user_id, { label: userOptionLabel(user), value: user.user_id }] as const),
        ).values(),
      ),
    [data],
  );
  return (
    <PaginatedMultiSelect
      inputId="shadow-eval-user"
      options={options}
      value={value}
      onValueChange={onChange}
      onSearchChange={setSearch}
      onLoadMore={() => void fetchNextPage()}
      hasNextPage={hasNextPage}
      isFetchingNextPage={isFetchingNextPage}
      isLoading={isPending}
      placeholder={t("costOptimization.shadowEval.start.users.placeholder")}
      emptyText={t("costOptimization.shadowEval.start.users.empty")}
      errorText={isError ? t("costOptimization.shadowEval.start.users.loadFailed") : undefined}
    />
  );
};

const RouterField: React.FC<{
  options: SearchSelectOption[];
  routerNames: string[];
  onChange: (names: string[]) => void;
  direction: ShadowEvalDirection;
}> = ({ options, routerNames, onChange, direction }) => {
  const { t } = useTranslation();
  return (
    <Field label={t("costOptimization.shadowEval.start.routers.label")}>
      <MultiSelect
        options={options}
        value={routerNames}
        onValueChange={onChange}
        placeholder={t("costOptimization.shadowEval.start.routers.placeholder")}
        emptyText={t("costOptimization.shadowEval.start.routers.empty")}
      />
      {routerNames.length > MAX_ROUTERS && (
        <p className="text-xs text-destructive">
          {t("costOptimization.shadowEval.start.routers.tooMany", { max: MAX_ROUTERS })}
        </p>
      )}
      {direction === "reverse" && routerNames.length > 1 && (
        <p className="text-xs text-destructive">{t("costOptimization.shadowEval.start.routers.reverseSingle")}</p>
      )}
      {direction === "forward" && routerNames.length > 1 && (
        <p className="text-xs text-muted-foreground">{t("costOptimization.shadowEval.start.routers.forwardShared")}</p>
      )}
    </Field>
  );
};

export const StartForm: React.FC = () => {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const form = useZodForm(shadowEvalSchema, { defaultValues: shadowEvalDefaults });
  const draft = { ...shadowEvalDefaults, ...useWatch({ control: form.control }) };
  const {
    apiKeyIds,
    teamIds,
    userIds,
    models,
    routerNames,
    direction,
    baselineModel,
    percentage,
    durationDays,
    judgeModel,
    maxBudget,
  } = draft;
  const directionOption = DIRECTION_OPTIONS.find((o) => o.value === direction);
  const durationOption = DURATION_OPTIONS.find((o) => o.value === durationDays);
  const { data: autoRouters } = useAutoRouters();
  const configuredGroups = usePlainModelGroups();
  const chatGroups = usePlainChatModelGroups();
  const chatDeployments = usePlainChatModelDeployments();
  const modelOptions = useMemo<SearchSelectOption[]>(
    () => [...configuredGroups].toSorted((a, b) => a.localeCompare(b)).map((name) => ({ label: name, value: name })),
    [configuredGroups],
  );
  const chatOptions = useMemo(
    () => modelOptions.filter((option) => chatGroups.has(option.value)),
    [modelOptions, chatGroups],
  );
  const chatAvailability = useMemo(
    () => buildModelAvailability(chatGroups, deploymentRefsFromModelInfo(chatDeployments)),
    [chatDeployments, chatGroups],
  );
  const recommendedJudgeModels = useMemo(
    () => new Set(RECOMMENDED_JUDGE_MODELS.flatMap((model) => resolveAvailableModels(model, chatAvailability))),
    [chatAvailability],
  );
  const judgeOptions = useMemo(
    () =>
      chatOptions.map((option) =>
        recommendedJudgeModels.has(option.value)
          ? { ...option, sublabel: t("costOptimization.shadowEval.start.judge.recommended") }
          : option,
      ),
    [chatOptions, recommendedJudgeModels, t],
  );
  const start = useStartShadowEval();

  const routerOptions = useMemo<SearchSelectOption[]>(() => {
    const names = new Set(
      (autoRouters ?? []).map((deployment) => deployment.model_name).filter((name): name is string => Boolean(name)),
    );
    return [...names].toSorted().map((name) => ({ label: name, value: name }));
  }, [autoRouters]);

  const { percentageValid, maxBudgetValid } = parseStartNumbers(draft);
  const valid = Boolean(accessToken) && shadowEvalSchema.safeParse(draft).success;
  const handleStart = () => {
    if (!valid || !judgeModel) return;
    start.mutate(buildStartBody(draft, judgeModel));
  };

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-foreground">
          {t("costOptimization.shadowEval.start.title")}
        </CardTitle>
        <p className="text-xs text-muted-foreground">{t(START_FORM_DESCRIPTION[direction])}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-3">
          <Field label={t("costOptimization.shadowEval.start.direction")}>
            <Select
              value={direction}
              onValueChange={(v: string | null) => form.setValue("direction", v === "reverse" ? "reverse" : "forward")}
            >
              <SelectTrigger className="w-full">
                <SelectValue>{directionOption && t(directionOption.labelKey)}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {DIRECTION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {t(option.labelKey)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label={t("costOptimization.shadowEval.start.keys.label")} htmlFor="shadow-eval-key">
            <KeySelect value={apiKeyIds} onChange={(value) => form.setValue("apiKeyIds", value)} />
          </Field>
          <Field label={t("costOptimization.shadowEval.start.teams.label")}>
            <TeamMultiSelect
              value={teamIds}
              onChange={(value) => form.setValue("teamIds", value)}
              placeholder={t("costOptimization.shadowEval.start.teams.placeholder")}
            />
          </Field>
          <Field label={t("costOptimization.shadowEval.start.users.label")} htmlFor="shadow-eval-user">
            <UserSelect value={userIds} onChange={(value) => form.setValue("userIds", value)} />
          </Field>
          {direction === "forward" && (
            <Field label={t("costOptimization.shadowEval.start.models.label")}>
              <MultiSelect
                options={modelOptions}
                value={models}
                onValueChange={(value) => form.setValue("models", value)}
                placeholder={t("costOptimization.shadowEval.start.models.placeholder")}
                emptyText={t("costOptimization.shadowEval.start.models.empty")}
              />
              {models.length > MAX_MODELS ? (
                <p className="text-xs text-destructive">
                  {t("costOptimization.shadowEval.start.models.tooMany", { max: MAX_MODELS })}
                </p>
              ) : (
                <p className="text-xs text-muted-foreground">{t("costOptimization.shadowEval.start.models.hint")}</p>
              )}
            </Field>
          )}
          <RouterField
            options={routerOptions}
            routerNames={routerNames}
            onChange={(value) => form.setValue("routerNames", value)}
            direction={direction}
          />
          <Field label={t("costOptimization.shadowEval.start.percentage.label")} htmlFor="shadow-eval-pct">
            <div className="flex items-center gap-2">
              <Input
                id="shadow-eval-pct"
                type="number"
                min={0.1}
                max={100}
                step={0.1}
                className="w-24"
                value={percentage}
                onChange={(e) => form.setValue("percentage", e.target.value)}
              />
              <span className="text-sm text-muted-foreground">
                {t("costOptimization.shadowEval.start.percentage.suffix")}
              </span>
            </div>
            <div>
              {percentage.trim() !== "" && !percentageValid && (
                <p className="text-xs text-destructive">{t("costOptimization.shadowEval.start.percentage.invalid")}</p>
              )}
            </div>
          </Field>
          <Field label={t("costOptimization.shadowEval.start.duration")}>
            <Select value={durationDays} onValueChange={(v: string | null) => form.setValue("durationDays", v ?? "7")}>
              <SelectTrigger className="w-full">
                <SelectValue>
                  {durationOption &&
                    t("costOptimization.shadowEval.start.durationDays", { count: durationOption.days })}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {DURATION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {t("costOptimization.shadowEval.start.durationDays", { count: option.days })}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label={t("costOptimization.shadowEval.start.budget.label")}>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">$</span>
              <Input
                type="number"
                min={0.01}
                max={10000}
                step={0.01}
                className="w-24"
                value={maxBudget}
                onChange={(e) => form.setValue("maxBudget", e.target.value)}
              />
              <span className="text-sm text-muted-foreground">
                {t("costOptimization.shadowEval.start.budget.suffix")}
              </span>
            </div>
            {maxBudget.trim() !== "" && !maxBudgetValid && (
              <p className="text-xs text-destructive">{t("costOptimization.shadowEval.start.budget.invalid")}</p>
            )}
          </Field>
          {direction === "reverse" && (
            <Field label={t("costOptimization.shadowEval.start.baseline.label")}>
              <SearchSelect
                options={chatOptions}
                value={baselineModel}
                onValueChange={(value) => form.setValue("baselineModel", value)}
                placeholder={t("costOptimization.shadowEval.start.baseline.placeholder")}
                emptyText={t("costOptimization.shadowEval.start.noChatModels")}
              />
            </Field>
          )}
          <Field label={t("costOptimization.shadowEval.start.judge.label")} className="sm:col-span-2">
            <SearchSelect
              options={judgeOptions}
              value={judgeModel}
              onValueChange={(value) => form.setValue("judgeModel", value)}
              placeholder={t("costOptimization.shadowEval.start.judge.placeholder")}
              emptyText={t("costOptimization.shadowEval.start.noChatModels")}
            />
          </Field>
        </div>
        <Button disabled={!valid || start.isPending} onClick={handleStart}>
          {start.isPending
            ? t("costOptimization.shadowEval.start.starting")
            : t("costOptimization.shadowEval.start.submit")}
        </Button>
      </CardContent>
    </Card>
  );
};
