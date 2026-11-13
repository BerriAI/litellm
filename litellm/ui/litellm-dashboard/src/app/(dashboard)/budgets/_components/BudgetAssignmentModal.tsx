import { useId, useState } from "react";
import { useTranslation } from "react-i18next";
import useAuthorized from "../../hooks/useAuthorized";
import { useBudgetAssignment } from "../../hooks/budgets/useBudgetAssignment";
import type { BudgetSubject } from "../../hooks/budgets/budgetProfileTypes";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import BudgetAssignmentEditor from "./BudgetAssignmentEditor";

function SubjectAssignment({ subject }: { subject: BudgetSubject }) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const { query, update } = useBudgetAssignment(subject);
  if (query.isPending)
    return (
      <p role="status" className="text-sm">
        {t("budgetProfiles.loading")}
      </p>
    );
  if (query.isError && !query.data)
    return (
      <p role="alert" className="text-sm text-destructive">
        {query.error.message}
      </p>
    );
  return (
    <BudgetAssignmentEditor
      assignment={query.data}
      accessToken={accessToken}
      save={(payload) => update.mutateAsync(payload)}
      reload={async () => (await query.refetch()).data}
    />
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
  const id = useId();
  const [type, setType] = useState<BudgetSubject["type"]>("user");
  const [subjectId, setSubjectId] = useState("");
  const [selected, setSelected] = useState<BudgetSubject | null>(null);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t("budgetProfiles.manage")}</DialogTitle>
          <DialogDescription>{t("budgetProfiles.assignmentHelp")}</DialogDescription>
        </DialogHeader>
        <form
          className="flex items-end gap-3 border-b pb-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (subjectId.trim()) setSelected({ type, id: subjectId.trim() });
          }}
        >
          <Field className="w-44 shrink-0">
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
          <Field className="min-w-0">
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
          <Button type="submit" variant="outline" disabled={!subjectId.trim()}>
            {t("budgetProfiles.load")}
          </Button>
        </form>
        {open && selected && <SubjectAssignment key={`${selected.type}:${selected.id}`} subject={selected} />}
      </DialogContent>
    </Dialog>
  );
}
