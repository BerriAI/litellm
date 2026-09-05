"use client";

import React, { useMemo, useState } from "react";

import { useInfiniteKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import { useInfiniteUsers } from "@/app/(dashboard)/hooks/users/useUsers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useAutoRouters, usePlainModelGroups } from "@/app/(dashboard)/hooks/models/useModels";
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

import { useStartShadowEval, type ShadowEvalJob } from "./useShadowEval";

type ShadowEvalDirection = ShadowEvalJob["direction"];

const MAX_ROUTERS = 4;

const RECOMMENDED_JUDGE_MODELS = ["anthropic/claude-sonnet-5", "openai/gpt-4o", "gemini/gemini-2.5-pro"] as const;

interface CostMapEntry {
  litellm_provider?: string;
  mode?: string;
}

const useChatModelNames = (): string[] => {
  const { data: costMap } = useModelCostMap();
  return useMemo(() => {
    if (!costMap) return [];
    const chatModels = Object.entries(costMap as Record<string, CostMapEntry>)
      .filter(([, value]) => value?.mode === "chat" && value?.litellm_provider)
      .map(([key, value]) => (key.startsWith(`${value.litellm_provider}/`) ? key : `${value.litellm_provider}/${key}`));
    return [...new Set(chatModels)].toSorted((a, b) => a.localeCompare(b));
  }, [costMap]);
};

const useJudgeModelOptions = (): SearchSelectOption[] => {
  const chatModels = useChatModelNames();
  return useMemo(() => {
    const pinned: SearchSelectOption[] = RECOMMENDED_JUDGE_MODELS.map((model) => ({
      label: model,
      value: model,
      sublabel: "Recommended",
    }));
    const pinnedNames = new Set<string>(RECOMMENDED_JUDGE_MODELS);
    const rest = chatModels.filter((model) => !pinnedNames.has(model)).map((model) => ({ label: model, value: model }));
    return [...pinned, ...rest];
  }, [chatModels]);
};

const useBaselineModelOptions = (): SearchSelectOption[] => {
  const configuredGroups = usePlainModelGroups();
  const chatModels = useChatModelNames();
  return useMemo(() => {
    const configured = [...configuredGroups]
      .toSorted((a, b) => a.localeCompare(b))
      .map((model) => ({ label: model, value: model, sublabel: "Configured on this gateway" }));
    const rest = chatModels
      .filter((model) => !configuredGroups.has(model))
      .map((model) => ({ label: model, value: model }));
    return [...configured, ...rest];
  }, [configuredGroups, chatModels]);
};

const DIRECTION_OPTIONS: readonly { value: ShadowEvalDirection; label: string }[] = [
  { value: "forward", label: "Adoption check: key's traffic vs the router" },
  { value: "reverse", label: "Regression check: router's picks vs a baseline" },
] as const;

const START_FORM_DESCRIPTION: Record<ShadowEvalDirection, string> = {
  forward:
    "Duplicates a sampled slice of the selected targets' traffic (keys, teams, or users) through the auto-router and has an LLM judge compare both answers blind. Each target gets its own spend budget. The router's answers are never served to users; judge calls bill to the sampled traffic's own identity.",
  reverse:
    "Duplicates a sampled slice of the traffic the auto-router already serves against a fixed baseline model and has an LLM judge compare both answers blind. Each target gets its own spend budget. The baseline's answers are never served to users; judge calls bill to the sampled traffic's own identity.",
};

const DURATION_OPTIONS = [
  { value: "1", label: "1 day" },
  { value: "3", label: "3 days" },
  { value: "7", label: "7 days" },
  { value: "14", label: "14 days" },
  { value: "30", label: "30 days" },
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
      placeholder="Search keys by alias"
      emptyText="No matching keys"
      errorText={isError ? "Keys could not be loaded. Refresh the page to retry." : undefined}
    />
  );
};

const UserSelect: React.FC<{ value: string[]; onChange: (ids: string[]) => void }> = ({ value, onChange }) => {
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
      placeholder="Search users by email"
      emptyText="No matching users"
      errorText={isError ? "Users could not be loaded. Refresh the page to retry." : undefined}
    />
  );
};

const RouterField: React.FC<{
  options: SearchSelectOption[];
  routerNames: string[];
  onChange: (names: string[]) => void;
  direction: ShadowEvalDirection;
}> = ({ options, routerNames, onChange, direction }) => (
  <Field label="Auto-routers">
    <MultiSelect
      options={options}
      value={routerNames}
      onValueChange={onChange}
      placeholder="Select up to 4 auto-routers"
      emptyText="No auto-routers configured"
    />
    {routerNames.length > MAX_ROUTERS && (
      <p className="text-xs text-destructive">Pick at most {MAX_ROUTERS} auto-routers</p>
    )}
    {direction === "reverse" && routerNames.length > 1 && (
      <p className="text-xs text-destructive">A regression check compares one router to its baseline</p>
    )}
    {direction === "forward" && routerNames.length > 1 && (
      <p className="text-xs text-muted-foreground">
        Every router sees the same sampled requests, judged against the same live responses
      </p>
    )}
  </Field>
);

interface StartFormValidityInputs {
  accessToken: string | null | undefined;
  apiKeyIds: string[];
  teamIds: string[];
  userIds: string[];
  routerNames: string[];
  direction: ShadowEvalDirection;
  baselineModel: string;
  judgeModel: string;
  percentage: string;
  maxBudget: string;
}

const startFormValidity = (inputs: StartFormValidityInputs) => {
  const parsedPct = Number.parseFloat(inputs.percentage);
  const percentageValid = parsedPct >= 0.1 && parsedPct <= 100;
  const parsedMaxBudget = Number.parseFloat(inputs.maxBudget);
  const maxBudgetValid = parsedMaxBudget >= 0.01 && parsedMaxBudget <= 10000;
  const baselinePicked = inputs.direction === "forward" || inputs.baselineModel !== "";
  const targetsPicked = inputs.apiKeyIds.length + inputs.teamIds.length + inputs.userIds.length > 0;
  const routerCountValid = inputs.routerNames.length >= 1 && inputs.routerNames.length <= MAX_ROUTERS;
  const routersMatchDirection = inputs.direction === "forward" || inputs.routerNames.length === 1;
  const routersValid = routerCountValid && routersMatchDirection;
  const modelsPicked = routersValid && inputs.judgeModel !== "" && baselinePicked;
  const filled = targetsPicked && modelsPicked;
  const boundsValid = percentageValid && maxBudgetValid;
  const valid = Boolean(inputs.accessToken) && filled && boundsValid;
  return { parsedPct, parsedMaxBudget, percentageValid, maxBudgetValid, valid };
};

interface StartBodyInputs {
  apiKeyIds: string[];
  teamIds: string[];
  userIds: string[];
  routerNames: string[];
  direction: ShadowEvalDirection;
  baselineModel: string;
  shadowPercentage: number;
  durationDays: number;
  maxBudget: number;
  judgeModel: string;
}

const buildStartBody = (inputs: StartBodyInputs) => ({
  api_key_ids: inputs.apiKeyIds,
  team_ids: inputs.teamIds,
  user_ids: inputs.userIds,
  router_names: inputs.routerNames,
  direction: inputs.direction,
  ...(inputs.direction === "reverse" ? { baseline_model: inputs.baselineModel } : {}),
  shadow_percentage: inputs.shadowPercentage,
  duration_days: inputs.durationDays,
  max_budget: inputs.maxBudget,
  judge_model: inputs.judgeModel,
});

export const StartForm: React.FC = () => {
  const { accessToken } = useAuthorized();
  const [apiKeyIds, setApiKeyIds] = useState<string[]>([]);
  const [teamIds, setTeamIds] = useState<string[]>([]);
  const [userIds, setUserIds] = useState<string[]>([]);
  const [routerNames, setRouterNames] = useState<string[]>([]);
  const [direction, setDirection] = useState<ShadowEvalDirection>("forward");
  const [baselineModel, setBaselineModel] = useState("");
  const [percentage, setPercentage] = useState("10");
  const [durationDays, setDurationDays] = useState("7");
  const [judgeModel, setJudgeModel] = useState("");
  const [maxBudget, setMaxBudget] = useState("10");
  const { data: autoRouters } = useAutoRouters();
  const judgeModelOptions = useJudgeModelOptions();
  const baselineModelOptions = useBaselineModelOptions();
  const start = useStartShadowEval();

  const routerOptions = useMemo<SearchSelectOption[]>(() => {
    const names = new Set(
      (autoRouters ?? []).map((deployment) => deployment.model_name).filter((name): name is string => Boolean(name)),
    );
    return [...names].toSorted().map((name) => ({ label: name, value: name }));
  }, [autoRouters]);

  const validityInputs: StartFormValidityInputs = {
    accessToken,
    apiKeyIds,
    teamIds,
    userIds,
    routerNames,
    direction,
    baselineModel,
    judgeModel,
    percentage,
    maxBudget,
  };
  const { parsedPct, parsedMaxBudget, percentageValid, maxBudgetValid, valid } = startFormValidity(validityInputs);
  const handleStart = () => {
    const bodyInputs: StartBodyInputs = {
      apiKeyIds,
      teamIds,
      userIds,
      routerNames,
      direction,
      baselineModel,
      shadowPercentage: parsedPct,
      durationDays: Number.parseInt(durationDays, 10),
      maxBudget: parsedMaxBudget,
      judgeModel,
    };
    start.mutate(buildStartBody(bodyInputs));
  };

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-foreground">Start a shadow eval</CardTitle>
        <p className="text-xs text-muted-foreground">{START_FORM_DESCRIPTION[direction]}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-3">
          <Field label="Direction">
            <Select
              value={direction}
              onValueChange={(v: string | null) => setDirection(v === "reverse" ? "reverse" : "forward")}
            >
              <SelectTrigger className="w-full">
                <SelectValue>{DIRECTION_OPTIONS.find((o) => o.value === direction)?.label}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {DIRECTION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label="Keys to shadow" htmlFor="shadow-eval-key">
            <KeySelect value={apiKeyIds} onChange={setApiKeyIds} />
          </Field>
          <Field label="Teams to shadow">
            <TeamMultiSelect value={teamIds} onChange={setTeamIds} placeholder="Search teams by alias" />
          </Field>
          <Field label="Users to shadow" htmlFor="shadow-eval-user">
            <UserSelect value={userIds} onChange={setUserIds} />
          </Field>
          <RouterField
            options={routerOptions}
            routerNames={routerNames}
            onChange={setRouterNames}
            direction={direction}
          />
          <Field label="Traffic sampled" htmlFor="shadow-eval-pct">
            <div className="flex items-center gap-2">
              <Input
                id="shadow-eval-pct"
                type="number"
                min={0.1}
                max={100}
                step={0.1}
                className="w-24"
                value={percentage}
                onChange={(e) => setPercentage(e.target.value)}
              />
              <span className="text-sm text-muted-foreground">% of traffic</span>
            </div>
            <div>
              {percentage.trim() !== "" && !percentageValid && (
                <p className="text-xs text-destructive">Enter a value from 0.1 to 100</p>
              )}
            </div>
          </Field>
          <Field label="Duration">
            <Select value={durationDays} onValueChange={(v: string | null) => setDurationDays(v ?? "7")}>
              <SelectTrigger className="w-full">
                <SelectValue>{DURATION_OPTIONS.find((o) => o.value === durationDays)?.label}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {DURATION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label="Spend budget">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">$</span>
              <Input
                type="number"
                min={0.01}
                max={10000}
                step={0.01}
                className="w-24"
                value={maxBudget}
                onChange={(e) => setMaxBudget(e.target.value)}
              />
              <span className="text-sm text-muted-foreground">max shadow + judge spend, per target</span>
            </div>
            {maxBudget.trim() !== "" && !maxBudgetValid && (
              <p className="text-xs text-destructive">Enter a value from 0.01 to 10000</p>
            )}
          </Field>
          {direction === "reverse" && (
            <Field label="Baseline model">
              <SearchSelect
                options={baselineModelOptions}
                value={baselineModel}
                onValueChange={setBaselineModel}
                placeholder="Select a baseline model"
                emptyText="No chat models available"
              />
            </Field>
          )}
          <Field label="Judge model" className="sm:col-span-2">
            <SearchSelect
              options={judgeModelOptions}
              value={judgeModel}
              onValueChange={setJudgeModel}
              placeholder="Select a judge model"
              emptyText="No chat models available"
            />
          </Field>
        </div>
        <Button disabled={!valid || start.isPending} onClick={handleStart}>
          {start.isPending ? "Starting..." : "Start shadow eval"}
        </Button>
      </CardContent>
    </Card>
  );
};
