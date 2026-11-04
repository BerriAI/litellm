"use client";

import React, { useEffect } from "react";
import { z } from "zod";
import { formatDate } from "@/components/networking";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
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
import type { SubscriptionAccountUsage, SubscriptionFeeFormValues } from "./types";

const feeShape = {
  monthly_fee: z
    .string()
    .min(1, "Enter a monthly fee")
    .refine((value) => Number(value) > 0, "Monthly fee must be greater than 0"),
  currency: z
    .string()
    .trim()
    .toUpperCase()
    .regex(/^[A-Z]{3}$/, "Currency must be a three-letter code"),
  billing_period_start: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "Pick a billing period start date"),
  label: z.string().trim(),
};

const feeSchema = z.object(feeShape);

type FeeFormInput = z.input<typeof feeSchema>;

const formInputFor = (account: SubscriptionAccountUsage | null): FeeFormInput => ({
  monthly_fee: account?.fee ? String(account.fee.monthly_fee) : "",
  currency: account?.fee?.currency ?? "USD",
  billing_period_start: account?.fee?.billing_period_start ?? formatDate(new Date()),
  label: account?.fee?.label ?? "",
});

interface SubscriptionFeeModalProps {
  visible: boolean;
  account: SubscriptionAccountUsage | null;
  onCancel: () => void;
  onSubmit: (values: SubscriptionFeeFormValues) => Promise<void>;
}

const SubscriptionFeeModal: React.FC<SubscriptionFeeModalProps> = ({ visible, account, onCancel, onSubmit }) => {
  const form = useZodForm(feeSchema, { defaultValues: formInputFor(account) });
  const { reset } = form;

  useEffect(() => {
    if (visible) reset(formInputFor(account));
  }, [visible, account, reset]);

  const handleFinish = async (values: z.output<typeof feeSchema>) => {
    const submitted: SubscriptionFeeFormValues = {
      monthly_fee: Number(values.monthly_fee),
      currency: values.currency,
      billing_period_start: values.billing_period_start,
      label: values.label === "" ? null : values.label,
    };
    await onSubmit(submitted);
  };

  const handleCancel = () => {
    reset(formInputFor(null));
    onCancel();
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>{account?.fee ? "Edit monthly fee" : "Set monthly fee"}</DialogTitle>
          <DialogDescription>
            {account ? `${account.custom_llm_provider} account ${account.account_id ?? ""}` : ""}
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleFinish)} noValidate>
          <FieldGroup>
            <FormField control={form.control} name="monthly_fee" label="Monthly fee">
              {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={0.01} min={0.01} />}
            </FormField>
            <FormField control={form.control} name="currency" label="Currency">
              {({ ref, ...field }) => <Input {...field} ref={ref} maxLength={3} />}
            </FormField>
            <FormField control={form.control} name="billing_period_start" label="Billing period start">
              {({ ref, ...field }) => <Input {...field} ref={ref} type="date" />}
            </FormField>
            <FormField control={form.control} name="label" label="Label (optional)">
              {({ ref, ...field }) => <Input {...field} ref={ref} />}
            </FormField>
          </FieldGroup>
          <DialogFooter className="mt-4">
            <Button type="button" variant="outline" onClick={handleCancel}>
              Cancel
            </Button>
            <Button type="submit" disabled={form.formState.isSubmitting}>
              Save
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default SubscriptionFeeModal;
