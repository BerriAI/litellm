import React, { useEffect } from "react";
import { useForm } from "react-hook-form";
import { Modal } from "antd";
import { CircleHelp } from "lucide-react";

import NumericalInput from "@/components/shared/numerical_input";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface EditableUser {
  user_id: string;
  user_email: string;
  user_role: string;
  spend: number | null;
  max_budget: number | null;
  budget_duration: string | null;
}

interface EditUserFormValues {
  user_email: string | undefined;
  user_id: string | undefined;
  user_role: string | undefined;
  spend: number | null | undefined;
  max_budget: number | string | null | undefined;
  budget_duration: string | null | undefined;
}

interface EditUserModalProps {
  visible: boolean;
  possibleUIRoles: null | Record<string, Record<string, string>>;
  onCancel: () => void;
  user: EditableUser | null;
  onSubmit: (data: EditUserFormValues) => void;
}

const SPEND_MIN = 0;

const toFormValues = (user: EditableUser | null): EditUserFormValues => ({
  user_email: user?.user_email,
  user_id: user?.user_id,
  user_role: user?.user_role,
  spend: user?.spend,
  max_budget: user?.max_budget,
  budget_duration: user?.budget_duration,
});

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const roleOption = (uiLabel: string, description: string): React.ReactNode => (
  <div className="flex">
    {uiLabel} <p className="ml-2 text-xs text-muted-foreground">{description}</p>
  </div>
);

const EditUserModal: React.FC<EditUserModalProps> = ({ visible, possibleUIRoles, onCancel, user, onSubmit }) => {
  const form = useForm<EditUserFormValues>({ defaultValues: toFormValues(user) });

  useEffect(() => {
    form.reset(toFormValues(user));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const handleCancel = async () => {
    form.reset(toFormValues(user));
    onCancel();
  };

  const handleEditSubmit = async (formValues: EditUserFormValues) => {
    onSubmit(formValues);
    form.reset(toFormValues(user));
    onCancel();
  };

  const clampSpendToMinimum = () => {
    const spend = form.getValues("spend");
    if (typeof spend === "number" && spend < SPEND_MIN) {
      form.setValue("spend", SPEND_MIN);
    }
  };

  if (!user) {
    return null;
  }

  const roleItems: Record<string, React.ReactNode> = Object.fromEntries(
    Object.entries(possibleUIRoles ?? {}).map(([role, { ui_label, description }]) => [
      role,
      roleOption(ui_label, description),
    ]),
  );

  return (
    <Modal open={visible} onCancel={handleCancel} footer={null} title={"Edit User " + user.user_id} width={1000}>
      <TooltipProvider>
        <form onSubmit={form.handleSubmit(handleEditSubmit)}>
          <FieldGroup className="mt-8">
            <FormField
              control={form.control}
              name="user_email"
              label={labelWithHint("User Email", "Email of the User")}
            >
              {({ ref, value, ...field }) => <Input {...field} ref={ref} value={value ?? ""} />}
            </FormField>

            <FormField control={form.control} name="user_role" label="User Role">
              {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                <Select
                  items={roleItems}
                  value={value ?? null}
                  onValueChange={(role: string | null) => onChange(role ?? undefined)}
                >
                  <SelectTrigger
                    id={id}
                    aria-invalid={ariaInvalid}
                    aria-describedby={ariaDescribedBy}
                    className="w-full"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(possibleUIRoles ?? {}).map(([role, { ui_label, description }]) => (
                      <SelectItem key={role} value={role} title={ui_label}>
                        {roleOption(ui_label, description)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>

            <FormField
              control={form.control}
              name="spend"
              label={labelWithHint("Spend (USD)", "(float) - Spend of all LLM calls completed by this user")}
              description="Across all keys (including keys with team_id)."
            >
              {({ ref, value, onChange, onBlur, ...field }) => (
                <Input
                  {...field}
                  ref={ref}
                  type="number"
                  min={SPEND_MIN}
                  step="any"
                  value={value ?? ""}
                  onChange={(event) => onChange(event.target.value === "" ? null : event.target.valueAsNumber)}
                  onBlur={() => {
                    onBlur();
                    clampSpendToMinimum();
                  }}
                />
              )}
            </FormField>

            <FormField
              control={form.control}
              name="max_budget"
              label={labelWithHint("User Budget (USD)", "(float) - Maximum budget of this user")}
              description="Maximum budget of this user."
            >
              {({ ref: _ref, value, ...field }) => (
                <NumericalInput {...field} min={0} step={0.01} value={value ?? ""} />
              )}
            </FormField>

            <FormField control={form.control} name="budget_duration" label="Reset Budget">
              {({ id, value, onChange }) => <BudgetDurationDropdown id={id} value={value} onChange={onChange} />}
            </FormField>
          </FieldGroup>

          <div className="mt-2.5 text-right">
            <Button type="submit">Save</Button>
          </div>

          <div className="mt-2.5 text-right">
            <Button type="submit">Save</Button>
          </div>
        </form>
      </TooltipProvider>
    </Modal>
  );
};

export default EditUserModal;
