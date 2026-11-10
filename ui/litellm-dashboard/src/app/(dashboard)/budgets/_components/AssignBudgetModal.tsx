import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState, type FormEvent } from "react";

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
  { value: "customer", label: "Customer / end user" },
  { value: "team", label: "Team / group" },
  { value: "user-agent", label: "User-agent pattern" },
  { value: "body", label: "Request field pattern" },
  { value: "header", label: "Request header pattern" },
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
  return (
    <Field>
      <FieldLabel htmlFor={`${id}-customer`}>Customer name or end-user ID</FieldLabel>
      <Input
        id={`${id}-customer`}
        list={`${id}-customers`}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        required
      />
      <datalist id={`${id}-customers`}>
        {customers.map((entry) => (
          <option key={entry.user_id} value={entry.user_id}>
            {entry.alias || entry.user_id}
          </option>
        ))}
      </datalist>
      <FieldDescription>
        Use an existing customer name or ID, or enter a new end-user ID. Each customer has its own spend counter
      </FieldDescription>
      {isError && <p role="alert">Could not load customers. Try reopening this dialog</p>}
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
  const options = (teams.data ?? []).map((team) => ({ value: team.team_id, label: team.team_alias || team.team_id }));
  return (
    <Field>
      <FieldLabel htmlFor={`${id}-team`}>Team</FieldLabel>
      <Select items={options} value={value} onValueChange={onChange}>
        <SelectTrigger id={`${id}-team`} className="w-full">
          <SelectValue placeholder="Select a team" />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <FieldDescription>
        Copies these budget limits to the team. Later changes to this budget do not update the team
      </FieldDescription>
      {teams.isError && <p role="alert">Could not load teams. Try reopening this dialog</p>}
    </Field>
  );
}

export default function AssignBudgetModal({ budget, accessToken, onClose }: AssignBudgetModalProps) {
  const id = useId();
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
  const isPattern = target !== "customer" && target !== "team";
  const assignedPatterns = (patterns.data?.rules ?? []).filter((rule) => rule.tag === budgetTagName(budget.budget_id));
  const assignment = useMutation({
    mutationFn: (value: BudgetAssignment) => assignBudget(apiClient, accessToken, budget, value),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: patternKeys });
      await queryClient.invalidateQueries({ queryKey: teamKeys.all });
      await queryClient.invalidateQueries({ queryKey: ["get", "/customer/list"] });
      toast.success("Budget assigned");
      onClose();
    },
    onError: (reason: Error) => setError(reason.message),
  });
  const removal = useMutation({
    mutationFn: async (rule: PatternMapping) => {
      const current = await loadPatternMappings(apiClient, accessToken);
      if (!current.editable) throw new Error("Pattern assignments are managed in the proxy configuration file");
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
        setError("More than one customer has this name. Enter the end-user ID instead");
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
    return patternsUnavailable || !patterns.data?.editable || budget.max_budget == null;
  };

  return (
    <Dialog open onOpenChange={(open) => !open && !pending && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Assign budget</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">Budget {budget.budget_id}</p>
        <form id={`${id}-form`} onSubmit={submit}>
          <FieldGroup className="gap-4">
            <Field>
              <FieldLabel htmlFor={`${id}-target`}>Assign to</FieldLabel>
              <Select
                items={TARGETS}
                value={target}
                onValueChange={(value) => {
                  if (value) setTarget(value);
                }}
              >
                <SelectTrigger id={`${id}-target`} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TARGETS.map((option) => (
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
                    <FieldLabel htmlFor={`${id}-field`}>
                      {target === "body" ? "Request field" : "Header name"}
                    </FieldLabel>
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
                  <FieldLabel htmlFor={`${id}-pattern`}>Pattern</FieldLabel>
                  <Input
                    id={`${id}-pattern`}
                    value={pattern}
                    onChange={(event) => setPattern(event.target.value)}
                    placeholder={target === "user-agent" ? "claude-code/*" : "engineering-*"}
                    required
                    maxLength={512}
                  />
                  <FieldDescription>
                    Case-sensitive wildcards. * matches any text and ? matches one character. Body fields must contain
                    strings. All matching requests share this budget
                  </FieldDescription>
                </Field>
                {budget.max_budget == null && (
                  <p role="alert">Set a spend limit on this budget before assigning patterns</p>
                )}
                {patterns.isError && <p role="alert">{patterns.error.message}</p>}
                {patterns.data?.editable === false && (
                  <p role="alert">Pattern assignments are managed in the proxy configuration file</p>
                )}
              </>
            )}
          </FieldGroup>
        </form>
        {assignedPatterns.length > 0 && (
          <section aria-label="Assigned patterns" className="space-y-2 border-t pt-4">
            <h3 className="text-sm font-medium">Assigned patterns</h3>
            {assignedPatterns.map((rule) => (
              <div key={JSON.stringify(rule)} className="flex items-center justify-between gap-3 text-sm">
                <span className="break-all">
                  {rule.source} {rule.field} matches {rule.pattern}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={pending || !patterns.data?.editable}
                  onClick={() => removal.mutate(rule)}
                  aria-label={`Remove ${rule.field} pattern ${rule.pattern}`}
                >
                  Remove
                </Button>
              </div>
            ))}
            <p className="text-xs text-muted-foreground">Removing a pattern keeps the budget and its recorded spend</p>
          </section>
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" disabled={pending} onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" form={`${id}-form`} disabled={pending || unavailable()}>
            {pending ? "Saving…" : "Assign budget"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
