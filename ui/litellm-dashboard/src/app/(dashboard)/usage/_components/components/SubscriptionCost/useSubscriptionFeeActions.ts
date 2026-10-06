import { useCallback, useState } from "react";
import {
  subscriptionAccountCreateCall,
  subscriptionAccountDeleteCall,
  subscriptionAccountUpdateCall,
} from "@/components/networking";
import { toast } from "@/lib/toast";
import type { SubscriptionAccountUsage, SubscriptionFeeFormValues } from "./types";

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

const saveFee = (
  accessToken: string,
  account: SubscriptionAccountUsage,
  values: SubscriptionFeeFormValues,
): Promise<unknown> | null => {
  if (account.fee) {
    return subscriptionAccountUpdateCall(accessToken, {
      subscription_account_id: account.fee.subscription_account_id,
      ...values,
    });
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
