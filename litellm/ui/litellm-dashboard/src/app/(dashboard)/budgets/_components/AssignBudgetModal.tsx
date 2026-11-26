import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";

import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { useCustomers, type EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { teamKeys, useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "@/lib/toast";

import {
  assignBudget,
  budgetTagName,
  loadPatternMappings,
  savePatternMappings,
  type BudgetAssignment,
  type PatternMapping,
} from "./budgetAssignments";

const TARGETS = [
  { value: "customer", labelKey: "budgets.assignModal.targets.customer" },
  { value: "team", labelKey: "budgets.assignModal.targets.team" },
  { value: "user-agent", labelKey: "budgets.assignModal.targets.userAgent" },
  { value: "body", labelKey: "budgets.assignModal.targets.body" },
  { value: "header", labelKey: "budgets.assignModal.targets.header" },
] as const;
type Target = (typeof TARGETS)[number]["value"];
const patternKeys = ["budget-pattern-assignments"];

interface AssignBudgetModalProps {
  budget: budgetItem;
  accessToken: string;
  onClose: () => void;
}

function CustomerField({
  id,
  value,
  onChange,
  customers,
  isError,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  customers: readonly EndUser[];
  isError: boolean;
}) {
  const { t } = useTranslation();
  const datalistId = `${id}-customers`;
  return (
    <Field>
      <FieldLabel htmlFor={`${id}-customer`}>{t("budgets.assignModal.customer.label")}</FieldLabel>
      <Input
        id={`${id}-customer`}
        list={datalistId}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        required
      />
      <datalist id={datalistId}>
        {customers.map((entry) => (
          <option key={entry.user_id} value={entry.user_id}>
            {entry.alias || entry.user_id}
          </option>
        ))}
      </datalist>
      <FieldDescription>{t("budgets.assignModal.customer.description")}</FieldDescription>
      {isError && <p role="alert">{t("budgets.assignModal.customer.loadFailed")}</p>}
    </Field>
  );
}

function TeamField({
  id,
  value,
  onChange,
  teams,
}: {
  id: string;
  value: string | null;
  onChange: (value: string | null) => void;
  teams: ReturnType<typeof useTeams>;
}) {
  const { t } = useTranslation();
  const options = (teams.data ?? []).map((team) => ({ value: team.team_id, label: team.team_alias || team.team_id }));
  return (
    <Field>
      <FieldLabel htmlFor={`${id}-team`}>{t("budgets.assignModal.team.label")}</FieldLabel>
      <Select items={options} value={value} onValueChange={onChange}>
        <SelectTrigger id={`${id}-team`} className="w-full">
          <SelectValue placeholder={t("budgets.assignModal.team.placeholder")} />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <FieldDescription>{t("budgets.assignModal.team.description")}</FieldDescription>
      {teams.isError && <p role="alert">{t("budgets.assignModal.team.loadFailed")}</p>}
    </Field>
  );
}

export default function AssignBudgetModal({ budget, accessToken, onClose }: AssignBudgetModalProps) {
  const { t } = useTranslation();
  const id = useId();
  const formId = `${id}-form`;
  const targets = TARGETS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }));
  const [target, setTarget] = useState<Target>("customer");
  const [customer, setCustomer] = useState("");
  const [teamId, setTeamId] = useState<string | null>(null);
  const [field, setField] = useState("");
  const [pattern, setPattern] = useState("");
  const [error, setError] = useState<string | null>(null);
  const customers = useCustomers(20, customer.trim());
  const customerList = customers.data?.pages.flat() ?? [];
  const teams = useTeams();
  const queryClient = useQueryClient();
  const patterns = useQuery({ queryKey: patternKeys, queryFn: () => loadPatternMappings(apiClient, accessToken) });
  const patternConfiguration = patterns.data ?? { rules: [], editable: false };
  const isPattern = target !== "customer" && target !== "team";
  const assignedPatterns = patternConfiguration.rules.filter((rule) => rule.tag === budgetTagName(budget.budget_id));
  const assignment = useMutation({
    mutationFn: (value: BudgetAssignment) => assignBudget(apiClient, accessToken, budget, value),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: patternKeys });
      await queryClient.invalidateQueries({ queryKey: teamKeys.all });
      await queryClient.invalidateQueries({ queryKey: ["get", "/customer/list"] });
      toast.success(t("budgets.assignModal.assigned"));
      onClose();
    },
    onError: (reason: Error) => setError(reason.message),
  });
  const removal = useMutation({
    mutationFn: async (rule: PatternMapping) => {
      const current = await loadPatternMappings(apiClient, accessToken);
      if (!current.editable) throw new Error(t("budgets.assignModal.patterns.configManaged"));
      await savePatternMappings(
        apiClient,
        accessToken,
        current.rules.filter((entry) => JSON.stringify(entry) !== JSON.stringify(rule)),
      );
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: patternKeys }),
    onError: (reason: Error) => setError(reason.message),
  });
  const pending = assignment.isPending || removal.isPending;

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    if (target === "customer") {
      const value = customer.trim();
      const exact = customerList.find((entry) => entry.user_id === value);
      const aliases = customerList.filter((entry) => entry.alias === value);
      if (!exact && aliases.length > 1) {
        setError(t("budgets.assignModal.customer.ambiguousName"));
        return;
      }
      const existing = exact ?? aliases[0];
      assignment.mutate({ kind: "customer", userId: existing?.user_id ?? value, existing: Boolean(existing) });
      return;
    }
    if (target === "team") {
      if (teamId) assignment.mutate({ kind: "team", teamId });
      return;
    }
    const patternAssignment: BudgetAssignment = {
      kind: "pattern",
      source: target === "body" ? "body" : "header",
      field: target === "user-agent" ? "user-agent" : field.trim(),
      pattern,
    };
    assignment.mutate(patternAssignment);
  };

  const unavailable = () => {
    if (target === "customer") return customers.isPending || customers.isError || !customer.trim();
    if (target === "team") return teams.isPending || teams.isError || !teamId;
    const patternsUnavailable = patterns.isPending || patterns.isError;
    return patternsUnavailable || !patternConfiguration.editable || budget.max_budget == null;
  };

  return (
    <Dialog open onOpenChange={(open) => !open && !pending && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("budgets.assignModal.title")}</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">{t("budgets.assignModal.budget", { id: budget.budget_id })}</p>
        <form id={formId} onSubmit={submit}>
          <FieldGroup className="gap-4">
            <Field>
              <FieldLabel htmlFor={`${id}-target`}>{t("budgets.assignModal.target")}</FieldLabel>
              <Select
                items={targets}
                value={target}
                onValueChange={(value) => {
                  if (value) setTarget(value);
                }}
              >
                <SelectTrigger id={`${id}-target`} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {targets.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            {target === "customer" && (
              <CustomerField
                id={id}
                value={customer}
                onChange={setCustomer}
                customers={customerList}
                isError={customers.isError}
              />
            )}
            {target === "team" && <TeamField id={id} value={teamId} onChange={setTeamId} teams={teams} />}
            {isPattern && (
              <>
                {target !== "user-agent" && (
                  <Field>
                    <FieldLabel htmlFor={`${id}-field`}>{t(`budgets.assignModal.fields.${target}`)}</FieldLabel>
                    <Input
                      id={`${id}-field`}
                      value={field}
                      onChange={(event) => setField(event.target.value)}
                      placeholder={target === "body" ? "metadata.group" : "x-client-name"}
                      required
                      maxLength={256}
                    />
                  </Field>
                )}
                <Field>
                  <FieldLabel htmlFor={`${id}-pattern`}>{t("budgets.assignModal.patterns.label")}</FieldLabel>
                  <Input
                    id={`${id}-pattern`}
                    value={pattern}
                    onChange={(event) => setPattern(event.target.value)}
                    placeholder={target === "user-agent" ? "claude-code/*" : "engineering-*"}
                    required
                    maxLength={512}
                  />
                  <FieldDescription>{t("budgets.assignModal.patterns.description")}</FieldDescription>
                </Field>
                {budget.max_budget == null && (
                  <p role="alert">{t("budgets.assignModal.patterns.spendLimitRequired")}</p>
                )}
                {patterns.isError && <p role="alert">{patterns.error.message}</p>}
                {patterns.data?.editable === false && (
                  <p role="alert">{t("budgets.assignModal.patterns.configManaged")}</p>
                )}
              </>
            )}
          </FieldGroup>
        </form>
        {assignedPatterns.length > 0 && (
          <section aria-label={t("budgets.assignModal.patterns.assigned")} className="space-y-2 border-t pt-4">
            <h3 className="text-sm font-medium">{t("budgets.assignModal.patterns.assigned")}</h3>
            {assignedPatterns.map((rule) => (
              <div key={JSON.stringify(rule)} className="flex items-center justify-between gap-3 text-sm">
                <span className="break-all">{t("budgets.assignModal.patterns.match", rule)}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={pending || !patternConfiguration.editable}
                  onClick={() => removal.mutate(rule)}
                  aria-label={t("budgets.assignModal.patterns.removeLabel", rule)}
                >
                  {t("budgets.assignModal.patterns.remove")}
                </Button>
              </div>
            ))}
            <p className="text-xs text-muted-foreground">{t("budgets.assignModal.patterns.removeDescription")}</p>
          </section>
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" disabled={pending} onClick={onClose}>
            {t("budgets.assignModal.cancel")}
          </Button>
          <Button type="submit" form={formId} disabled={pending || unavailable()}>
            {pending ? t("budgets.assignModal.saving") : t("budgets.assignModal.title")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
