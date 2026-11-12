"use client";

import React, { useEffect } from "react";
import { useTranslation } from "react-i18next";
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
import i18n from "@/i18n/i18n";
import { useZodForm } from "@/lib/forms/useZodForm";
import type { SubscriptionAccountUsage, SubscriptionFeeFormValues } from "./types";

const feeShape = {
  monthly_fee: z
    .string()
    .min(1, { error: () => i18n.t("usage.subscriptionCost.feeModal.errors.feeRequired") })
    .refine((value) => Number(value) > 0, {
      error: () => i18n.t("usage.subscriptionCost.feeModal.errors.feePositive"),
    }),
  currency: z
    .string()
    .trim()
    .toUpperCase()
    .regex(/^[A-Z]{3}$/, { error: () => i18n.t("usage.subscriptionCost.feeModal.errors.currencyCode") }),
  billing_period_start: z
    .string()
    .regex(/^\d{4}-\d{2}-\d{2}$/, { error: () => i18n.t("usage.subscriptionCost.feeModal.errors.periodStart") }),
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
  const { t } = useTranslation();
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
          <DialogTitle>
            {account?.fee
              ? t("usage.subscriptionCost.feeModal.editTitle")
              : t("usage.subscriptionCost.feeModal.setTitle")}
          </DialogTitle>
          <DialogDescription>
            {account
              ? t("usage.subscriptionCost.feeModal.description", {
                  provider: account.custom_llm_provider,
                  accountId: account.account_id ?? "",
                })
              : ""}
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleFinish)} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="monthly_fee"
              label={t("usage.subscriptionCost.feeModal.fields.monthlyFee")}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} type="number" step={0.01} min={0.01} />}
            </FormField>
            <FormField
              control={form.control}
              name="currency"
              label={t("usage.subscriptionCost.feeModal.fields.currency")}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} maxLength={3} />}
            </FormField>
            <FormField
              control={form.control}
              name="billing_period_start"
              label={t("usage.subscriptionCost.feeModal.fields.billingPeriodStart")}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} type="date" />}
            </FormField>
            <FormField control={form.control} name="label" label={t("usage.subscriptionCost.feeModal.fields.label")}>
              {({ ref, ...field }) => <Input {...field} ref={ref} />}
            </FormField>
          </FieldGroup>
          <DialogFooter className="mt-4">
            <Button type="button" variant="outline" onClick={handleCancel}>
              {t("usage.subscriptionCost.feeModal.cancel")}
            </Button>
            <Button type="submit" disabled={form.formState.isSubmitting}>
              {t("usage.subscriptionCost.feeModal.save")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default SubscriptionFeeModal;
