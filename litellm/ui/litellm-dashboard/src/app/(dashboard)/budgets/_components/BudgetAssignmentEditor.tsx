import { useEffect, useState } from "react";
import { useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import type { BudgetAssignment, BudgetAssignmentPatch } from "../../hooks/budgets/budgetProfileTypes";
import { useBudgetOptions } from "../../hooks/budgets/useBudgetOptions";
import { FormField } from "@/components/shared/form/FormField";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { FieldDescription, FieldGroup, FieldLegend, FieldSet } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useZodForm } from "@/lib/forms/useZodForm";
import { ApiError } from "@/lib/http/client";
import { toast } from "@/lib/toast";
import { assignmentPayload, assignmentValues } from "./budgetAssignmentPayload";

const shape = {
  profile: z.string(),
  monthly: z.number().nonnegative().nullable(),
  annual: z.number().nonnegative().nullable(),
  policy: z.enum(["inherit", "pause_allow_increase", "pause_until_reset"]),
  increase: z.number().nonnegative().nullable(),
  increaseExpiry: z.string(),
  increaseDuration: z.enum(["1mo", "12mo"]),
  overrideExpiry: z.string(),
  clearOverride: z.boolean(),
};
const schema = z.object(shape);
const PERSONAL_CAP_FIELDS = ["monthly", "annual"] as const;
const hasExpired = (expiry: string | null | undefined, now: number) => expiry != null && Date.parse(expiry) <= now;

interface Props {
  assignment: BudgetAssignment;
  accessToken: string | null;
  save: (payload: BudgetAssignmentPatch) => Promise<BudgetAssignment>;
  reload: () => Promise<BudgetAssignment | undefined>;
}

export default function BudgetAssignmentEditor({ assignment, accessToken, save, reload }: Props) {
  const { t } = useTranslation();
  const [baseline, setBaseline] = useState(assignment);
  const [error, setError] = useState<Error | null>(null);
  const [now, setNow] = useState(Date.now);
  const form = useZodForm(schema, { defaultValues: assignmentValues(assignment) });
  const { isDirty, isSubmitting } = form.formState;
  const clearOverride = useWatch({ control: form.control, name: "clearOverride" });
  const budgets = useBudgetOptions(accessToken);
  const profiles = (budgets.data ?? []).filter((budget) => budget.is_profile);

  if (!isDirty && baseline !== assignment) setBaseline(assignment);

  useEffect(() => form.reset(assignmentValues(baseline)), [baseline, form]);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(timer);
  }, []);

  const accept = (next: BudgetAssignment) => {
    setBaseline(next);
    form.reset(assignmentValues(next));
    setError(null);
  };
  const onSubmit = form.handleSubmit(async (values) => {
    const result = assignmentPayload(values, baseline);
    if (!result.ok) {
      form.setError(result.field, { message: t(`budgetProfiles.${result.error}`) });
      return;
    }
    try {
      accept(await save(result.payload));
      toast.success(t("budgetProfiles.saved"));
    } catch (caught) {
      setError(caught instanceof Error ? caught : new Error(String(caught)));
    }
  });
  const expiredOverride = hasExpired(baseline.budget_override_expires_at, now);
  const expiredIncrease = hasExpired(baseline.budget_override?.temp_budget_expiry, now);
  const conflict = error instanceof ApiError && error.status === 409;

  return (
    <form onSubmit={onSubmit} className="space-y-5" noValidate>
      <fieldset disabled={isSubmitting} className="space-y-5">
        <FormField
          control={form.control}
          name="profile"
          label={t("budgetProfiles.assignedProfile")}
          description={t("budgetProfiles.personalOnly")}
        >
          {({ id, value, onChange }) => (
            <SearchSelect
              inputId={id}
              value={value}
              onValueChange={(next) => onChange(next ?? "")}
              options={profiles.map((profile) => ({ label: profile.budget_id, value: profile.budget_id }))}
              placeholder={t("budgetProfiles.noProfile")}
              emptyText={t("budgetProfiles.noProfiles")}
              disabled={budgets.isPending || budgets.isError || isSubmitting}
            />
          )}
        </FormField>
        {budgets.isError && (
          <p role="alert" className="text-sm text-destructive">
            {budgets.error.message}
          </p>
        )}
        <FieldSet className="rounded-lg border p-4">
          <FieldLegend>{t("budgetProfiles.personalOverrides")}</FieldLegend>
          <FieldDescription>{t("budgetProfiles.overrideHelp")}</FieldDescription>
          {expiredOverride && (
            <p role="status" className="text-sm text-amber-700">
              {t("budgetProfiles.overrideExpired")}
            </p>
          )}
          {clearOverride ? (
            <p role="status" className="text-sm">
              {t("budgetProfiles.clearPending")}
            </p>
          ) : (
            <FieldGroup className="gap-4">
              <div className="grid grid-cols-2 gap-3">
                {PERSONAL_CAP_FIELDS.map((name) => (
                  <FormField
                    key={name}
                    control={form.control}
                    name={name}
                    label={t(name === "monthly" ? "budgetProfiles.personalMonthly" : "budgetProfiles.personalAnnual")}
                  >
                    {({ value, onChange, ...field }) => (
                      <Input
                        {...field}
                        type="number"
                        min={0}
                        step="any"
                        value={value ?? ""}
                        placeholder={t("budgetProfiles.inherit")}
                        onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
                      />
                    )}
                  </FormField>
                ))}
              </div>
              <FormField
                control={form.control}
                name="policy"
                label={t("budgetProfiles.capPolicy")}
                description={t("budgetProfiles.capPolicyHelp")}
              >
                {({ id, value, onChange }) => (
                  <Select
                    value={value}
                    onValueChange={onChange}
                    items={[
                      { value: "inherit", label: t("budgetProfiles.inherit") },
                      { value: "pause_until_reset", label: t("budgetProfiles.pauseUntilReset") },
                      { value: "pause_allow_increase", label: t("budgetProfiles.pauseAllowIncrease") },
                    ]}
                  >
                    <SelectTrigger id={id}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="inherit">{t("budgetProfiles.inherit")}</SelectItem>
                      <SelectItem value="pause_until_reset">{t("budgetProfiles.pauseUntilReset")}</SelectItem>
                      <SelectItem value="pause_allow_increase">{t("budgetProfiles.pauseAllowIncrease")}</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              </FormField>
              <FormField
                control={form.control}
                name="overrideExpiry"
                label={t("budgetProfiles.overrideExpires")}
                description={t("budgetProfiles.overrideExpiryHelp")}
              >
                {(field) => <Input {...field} type="datetime-local" />}
              </FormField>
              {expiredOverride && (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => form.setValue("overrideExpiry", "", { shouldDirty: true })}
                >
                  {t("budgetProfiles.reactivate")}
                </Button>
              )}
              <FieldSet className="gap-3 border-t pt-4">
                <FieldLegend>{t("budgetProfiles.temporaryIncrease")}</FieldLegend>
                <FieldDescription>{t("budgetProfiles.increaseHelp")}</FieldDescription>
                {expiredIncrease && (
                  <p role="status" className="text-sm text-amber-700">
                    {t("budgetProfiles.increaseExpired")}
                  </p>
                )}
                <div className="grid grid-cols-2 gap-3">
                  <FormField control={form.control} name="increase" label={t("budgetProfiles.increaseAmount")}>
                    {({ value, onChange, ...field }) => (
                      <Input
                        {...field}
                        type="number"
                        min={0}
                        step="any"
                        value={value ?? ""}
                        placeholder={t("budgetProfiles.noIncrease")}
                        onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
                      />
                    )}
                  </FormField>
                  <FormField control={form.control} name="increaseDuration" label={t("budgetProfiles.increaseApplies")}>
                    {({ id, value, onChange }) => (
                      <Select
                        value={value}
                        onValueChange={onChange}
                        items={[
                          { value: "1mo", label: t("budgetProfiles.monthly") },
                          { value: "12mo", label: t("budgetProfiles.annual") },
                        ]}
                      >
                        <SelectTrigger id={id}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1mo">{t("budgetProfiles.monthly")}</SelectItem>
                          <SelectItem value="12mo">{t("budgetProfiles.annual")}</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  </FormField>
                </div>
                <FormField control={form.control} name="increaseExpiry" label={t("budgetProfiles.increaseExpires")}>
                  {(field) => <Input {...field} type="datetime-local" />}
                </FormField>
              </FieldSet>
            </FieldGroup>
          )}
          <Button
            type="button"
            variant="outline"
            onClick={() => form.setValue("clearOverride", !clearOverride, { shouldDirty: true })}
          >
            {t(clearOverride ? "budgetProfiles.undoClear" : "budgetProfiles.clearOverrides")}
          </Button>
        </FieldSet>
      </fieldset>
      <section
        aria-label={t("budgetProfiles.effectiveLimits")}
        className="space-y-2 rounded-lg bg-muted/50 p-4 text-sm"
      >
        <p className="font-medium">{t("budgetProfiles.effectiveLimits")}</p>
        <FieldDescription>{t("budgetProfiles.effectiveHelp")}</FieldDescription>
        {assignment.windows.map((window) => (
          <div key={window.budget_duration} className="flex justify-between gap-4">
            <span>{t(window.budget_duration === "1mo" ? "budgetProfiles.monthly" : "budgetProfiles.annual")}</span>
            <span>
              {t("budgetProfiles.effectiveWindow", {
                amount: window.max_budget,
                reset: new Date(window.reset_at).toLocaleString(),
              })}
            </span>
          </div>
        ))}
        {assignment.windows.length === 0 && <p>{t("budgetProfiles.noLimits")}</p>}
      </section>
      {error && (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>{conflict ? t("budgetProfiles.conflict") : error.message}</p>
          {conflict && (
            <Button
              type="button"
              variant="outline"
              onClick={async () => {
                const next = await reload();
                if (next) accept(next);
              }}
            >
              {t("budgetProfiles.reload")}
            </Button>
          )}
        </div>
      )}
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm text-muted-foreground">
          {isDirty ? t("budgetProfiles.unsaved") : t("budgetProfiles.upToDate")}
        </span>
        <Button type="submit" disabled={!isDirty || isSubmitting || conflict}>
          {t("budgetProfiles.save")}
        </Button>
      </div>
    </form>
  );
}
