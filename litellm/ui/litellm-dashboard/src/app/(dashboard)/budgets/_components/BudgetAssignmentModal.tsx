import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { useEffect, useId, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { budgetKeys } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { apiClient } from "@/components/networking";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useZodForm } from "@/lib/forms/useZodForm";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";

type Assignment = components["schemas"]["BudgetAssignment"];
type Subject = { type: "user" | "customer_member"; id: string };

const assignmentShape = {
  profile: z.string(),
  monthly: z.number().nonnegative().nullable(),
  annual: z.number().nonnegative().nullable(),
  increase: z.number().nonnegative().nullable(),
  expiry: z.string(),
  duration: z.enum(["1mo", "12mo"]),
};
const NUMBER_FIELDS = ["monthly", "annual", "increase"] as const;
const schema = z.object(assignmentShape).refine((values) => values.increase == null || values.expiry !== "", {
  message: i18n.t("budgetProfiles.expiryRequired"),
  path: ["expiry"],
});

const defaults = (assignment: Assignment): z.input<typeof schema> => ({
  profile: assignment.budget_template_id ?? "",
  monthly:
    assignment.budget_override?.budget_limits?.find((limit) => limit.budget_duration === "1mo")?.max_budget ?? null,
  annual:
    assignment.budget_override?.budget_limits?.find((limit) => limit.budget_duration === "12mo")?.max_budget ?? null,
  increase: assignment.budget_override?.temp_budget_increase ?? null,
  expiry: assignment.budget_override?.temp_budget_expiry
    ? new Date(assignment.budget_override.temp_budget_expiry).toISOString().slice(0, 16)
    : "",
  duration: assignment.budget_override?.temp_budget_duration ?? "1mo",
});

function AssignmentEditor({
  assignment,
  onSave,
}: {
  assignment: Assignment;
  onSave: (payload: components["schemas"]["MemberPatch"]) => Promise<void>;
}) {
  const { t } = useTranslation();
  const form = useZodForm(schema, { defaultValues: defaults(assignment) });
  useEffect(() => form.reset(defaults(assignment)), [assignment, form]);
  const save = form.handleSubmit(async (values) => {
    const budgetLimits = [
      ...(values.monthly == null ? [] : [{ budget_duration: "1mo", max_budget: values.monthly }]),
      ...(values.annual == null ? [] : [{ budget_duration: "12mo", max_budget: values.annual }]),
    ];
    const overrideChanged = (["monthly", "annual", "increase", "expiry", "duration"] as const).some(
      (field) => form.formState.dirtyFields[field],
    );
    await onSave({
      ...(form.formState.dirtyFields.profile ? { budget_template_id: values.profile.trim() || null } : {}),
      ...(overrideChanged
        ? {
            budget_override: {
              ...assignment.budget_override,
              budget_limits: budgetLimits,
              temp_budget_increase: values.increase,
              temp_budget_expiry: values.increase == null ? null : new Date(`${values.expiry}Z`).toISOString(),
              temp_budget_duration: values.increase == null ? null : values.duration,
            },
          }
        : {}),
    });
  });
  return (
    <form onSubmit={save} className="mt-4 space-y-4">
      <FieldGroup className="gap-3">
        <FormField
          control={form.control}
          name="profile"
          label={t("budgetProfiles.profileId")}
          description={t("budgetProfiles.personalOnly")}
        >
          {(field) => <Input {...field} />}
        </FormField>
        {NUMBER_FIELDS.map((name) => (
          <FormField
            key={name}
            control={form.control}
            name={name}
            label={
              {
                monthly: t("budgetProfiles.personalMonthly"),
                annual: t("budgetProfiles.personalAnnual"),
                increase: t("budgetProfiles.temporaryIncrease"),
              }[name]
            }
          >
            {({ value, onChange, ...field }) => (
              <Input
                {...field}
                type="number"
                min={0}
                step={0.01}
                value={value ?? ""}
                placeholder={name === "increase" ? t("budgetProfiles.noIncrease") : t("budgetProfiles.inherit")}
                onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
              />
            )}
          </FormField>
        ))}
        <FormField control={form.control} name="duration" label={t("budgetProfiles.increaseApplies")}>
          {(field) => (
            <Select
              value={field.value}
              onValueChange={field.onChange}
              items={[
                { value: "1mo", label: t("budgetProfiles.monthlyLimit") },
                { value: "12mo", label: t("budgetProfiles.annualLimit") },
              ]}
            >
              <SelectTrigger id={field.id}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1mo">{t("budgetProfiles.monthlyLimit")}</SelectItem>
                <SelectItem value="12mo">{t("budgetProfiles.annualLimit")}</SelectItem>
              </SelectContent>
            </Select>
          )}
        </FormField>
        <FormField control={form.control} name="expiry" label={t("budgetProfiles.increaseExpires")}>
          {(field) => <Input {...field} type="datetime-local" />}
        </FormField>
        <FieldDescription>{t("budgetProfiles.overrideHelp")}</FieldDescription>
      </FieldGroup>
      <div className="rounded-lg border p-3 text-sm">
        <p className="mb-2 font-medium">{t("budgetProfiles.effectiveLimits")}</p>
        {assignment.windows.map((window) => (
          <p key={window.budget_duration}>
            {window.budget_duration === "1mo" ? t("budgetProfiles.monthly") : t("budgetProfiles.annual")} · $
            {window.max_budget.toFixed(2)}
          </p>
        ))}
        {assignment.windows.length === 0 && <p>{t("budgetProfiles.noLimits")}</p>}
      </div>
      <Button type="submit" disabled={form.formState.isSubmitting}>
        {t("budgetProfiles.save")}
      </Button>
    </form>
  );
}

export default function BudgetAssignmentModal({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useTranslation();
  const { accessToken, userRole } = useAuthorized();
  const queryClient = useQueryClient();
  const id = useId();
  const [type, setType] = useState<Subject["type"]>("user");
  const [subjectId, setSubjectId] = useState("");
  const [selected, setSelected] = useState<Subject | null>(null);
  const path = selected ? `/management/v1/budget_assignments/${selected.type}/${encodeURIComponent(selected.id)}` : "";
  const queryKey = ["budget-assignment", accessToken, userRole, selected?.type, selected?.id];
  const assignment = useQuery({
    queryKey,
    queryFn: ({ signal }) => apiClient.get<Assignment>(path, { accessToken, signal }),
    enabled: open && selected !== null && Boolean(accessToken),
  });
  const update = useMutation({
    mutationFn: (request: {
      path: string;
      queryKey: readonly unknown[];
      accessToken: typeof accessToken;
      body: components["schemas"]["MemberPatch"];
    }) => apiClient.patch<Assignment>(request.path, { accessToken: request.accessToken, body: request.body }),
    onSuccess: async (result, request) => {
      await queryClient.cancelQueries({ queryKey: request.queryKey, exact: true });
      queryClient.setQueryData(request.queryKey, result);
      await queryClient.invalidateQueries({ queryKey: budgetKeys.all });
      toast.success(t("budgetProfiles.saved"));
    },
    onError: (error) => toast.fromError(error),
  });
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("budgetProfiles.manage")}</DialogTitle>
        </DialogHeader>
        <FieldGroup className="gap-3">
          <Field>
            <FieldLabel htmlFor={`${id}-type`}>{t("budgetProfiles.personType")}</FieldLabel>
            <Select
              value={type}
              onValueChange={(value) => {
                if (value === "user" || value === "customer_member") {
                  setType(value);
                  setSelected(null);
                }
              }}
              items={[
                { value: "user", label: t("budgetProfiles.user") },
                { value: "customer_member", label: t("budgetProfiles.customerMember") },
              ]}
            >
              <SelectTrigger id={`${id}-type`}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="user">{t("budgetProfiles.user")}</SelectItem>
                <SelectItem value="customer_member">{t("budgetProfiles.customerMember")}</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field>
            <FieldLabel htmlFor={`${id}-subject`}>{t("budgetProfiles.personId")}</FieldLabel>
            <Input
              id={`${id}-subject`}
              value={subjectId}
              onChange={(event) => {
                setSubjectId(event.target.value);
                setSelected(null);
              }}
            />
          </Field>
          <Button
            variant="outline"
            disabled={!subjectId.trim()}
            onClick={() => setSelected({ type, id: subjectId.trim() })}
          >
            {t("budgetProfiles.load")}
          </Button>
        </FieldGroup>
        {assignment.isFetching && <p className="text-sm text-muted-foreground">{t("budgetProfiles.loading")}</p>}
        {assignment.error && (
          <p role="alert" className="text-sm text-destructive">
            {assignment.error.message}
          </p>
        )}
        {selected && assignment.data && (
          <AssignmentEditor
            key={`${selected.type}:${selected.id}`}
            assignment={assignment.data}
            onSave={async (body) => {
              const request = { path, queryKey, accessToken, body };
              await update.mutateAsync(request);
            }}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
