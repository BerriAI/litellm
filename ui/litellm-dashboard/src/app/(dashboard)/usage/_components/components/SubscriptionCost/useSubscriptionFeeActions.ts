import { useCallback, useState } from "react";
import {
  subscriptionAccountCreateCall,
  subscriptionAccountDeleteCall,
  subscriptionAccountUpdateCall,
} from "@/components/networking";
import { toast } from "@/lib/toast";
import type {
  SubscriptionAccountUpdateRequest,
  SubscriptionAccountUsage,
  SubscriptionFee,
  SubscriptionFeeFormValues,
} from "./types";

interface Options {
  accessToken: string | null;
  refetch: () => Promise<unknown>;
}

interface Result {
  feeModalAccount: SubscriptionAccountUsage | null;
  openFeeModal: (account: SubscriptionAccountUsage) => void;
  closeFeeModal: () => void;
  submitFee: (values: SubscriptionFeeFormValues) => Promise<void>;
  removeFee: (account: SubscriptionAccountUsage) => Promise<void>;
}

const changedFields = (fee: SubscriptionFee, values: SubscriptionFeeFormValues): SubscriptionAccountUpdateRequest => ({
  subscription_account_id: fee.subscription_account_id,
  ...(values.monthly_fee !== fee.monthly_fee && { monthly_fee: values.monthly_fee }),
  ...(values.currency !== fee.currency && { currency: values.currency }),
  ...(values.billing_period_start !== fee.billing_period_start && {
    billing_period_start: values.billing_period_start,
  }),
  ...(values.label !== fee.label && { label: values.label }),
});

const saveFee = (
  accessToken: string,
  account: SubscriptionAccountUsage,
  values: SubscriptionFeeFormValues,
): Promise<unknown> | null => {
  if (account.fee) {
    return subscriptionAccountUpdateCall(accessToken, changedFields(account.fee, values));
  }
  if (account.account_id === null) return null;
  return subscriptionAccountCreateCall(accessToken, {
    custom_llm_provider: account.custom_llm_provider,
    account_id: account.account_id,
    ...values,
  });
};

export function useSubscriptionFeeActions({ accessToken, refetch }: Options): Result {
  const [feeModalAccount, setFeeModalAccount] = useState<SubscriptionAccountUsage | null>(null);
  const closeFeeModal = useCallback(() => setFeeModalAccount(null), []);

  const submitFee = useCallback(
    async (values: SubscriptionFeeFormValues) => {
      if (!accessToken || feeModalAccount === null) return;
      const request = saveFee(accessToken, feeModalAccount, values);
      if (request === null) return;
      try {
        await request;
        toast.success("Monthly fee saved");
        setFeeModalAccount(null);
        await refetch();
      } catch (error) {
        toast.fromError(error);
      }
    },
    [accessToken, feeModalAccount, refetch],
  );

  const removeFee = useCallback(
    async (account: SubscriptionAccountUsage) => {
      if (!accessToken || account.fee === null) return;
      try {
        await subscriptionAccountDeleteCall(accessToken, account.fee.subscription_account_id);
        toast.success("Monthly fee removed");
        await refetch();
      } catch (error) {
        toast.fromError(error);
      }
    },
    [accessToken, refetch],
  );

  return { feeModalAccount, openFeeModal: setFeeModalAccount, closeFeeModal, submitFee, removeFee };
}
