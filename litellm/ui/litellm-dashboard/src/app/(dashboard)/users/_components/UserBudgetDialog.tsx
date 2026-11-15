import { useId } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { z } from "zod";

import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import { UserInfo, userUpdateUserCall } from "@/components/networking";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";

import { userBudgetSchema } from "./userBudgetForm";

interface UserBudgetDialogProps {
  user: Pick<UserInfo, "user_id" | "user_email" | "max_budget" | "budget_duration">;
  accessToken: string;
  onClose: () => void;
}

export default function UserBudgetDialog({ user, accessToken, onClose }: UserBudgetDialogProps) {
  const formId = useId();
  const queryClient = useQueryClient();
  const form = useZodForm(userBudgetSchema, {
    defaultValues: {
      unlimited: user.max_budget == null,
      max_budget: user.max_budget?.toString() ?? "",
      budget_duration: user.budget_duration ?? null,
    },
  });
  const unlimited = form.watch("unlimited");
  const { isSubmitting, dirtyFields } = form.formState;

  const save = async (values: z.output<typeof userBudgetSchema>) => {
    try {
      await userUpdateUserCall(
        accessToken,
        {
          user_id: user.user_id,
          max_budget: values.unlimited ? null : Number(values.max_budget),
          ...(dirtyFields.budget_duration ? { budget_duration: values.budget_duration } : {}),
        },
        null,
      );
      await queryClient.invalidateQueries({ queryKey: ["userList"] });
      toast.success("Budget updated");
      onClose();
    } catch (error) {
      toast.fromError(error);
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && !isSubmitting && onClose()}>
      <DialogContent showCloseButton={!isSubmitting}>
        <DialogHeader>
          <DialogTitle>Adjust budget</DialogTitle>
          <DialogDescription>{user.user_email || user.user_id}</DialogDescription>
        </DialogHeader>
        <form id={formId} onSubmit={form.handleSubmit(save)} noValidate>
          <fieldset disabled={isSubmitting}>
            <FieldGroup>
              <FormField control={form.control} name="unlimited" label="Unlimited budget" orientation="horizontal">
                {({ id, value, onChange }) => <Checkbox id={id} checked={value} onCheckedChange={onChange} />}
              </FormField>
              <FormField control={form.control} name="max_budget" label="Max Budget (USD)">
                {({ ref, value, onChange, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    type="number"
                    min={0}
                    step="any"
                    value={value}
                    onChange={(event) => onChange(event.target.value)}
                    disabled={unlimited || isSubmitting}
                  />
                )}
              </FormField>
              <FormField control={form.control} name="budget_duration" label="Reset Budget">
                {({ id, value, onChange }) => <BudgetDurationDropdown id={id} value={value} onChange={onChange} />}
              </FormField>
            </FieldGroup>
          </fieldset>
        </form>
        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button type="submit" form={formId} disabled={isSubmitting}>
            {isSubmitting ? "Saving..." : "Save budget"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
