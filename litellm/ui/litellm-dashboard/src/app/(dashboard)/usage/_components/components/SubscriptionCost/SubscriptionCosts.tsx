import React from "react";
import { useSubscriptionUsage, type SubscriptionUsageOptions } from "../../hooks/useSubscriptionUsage";
import SubscriptionCostCard from "./SubscriptionCostCard";
import SubscriptionFeeModal from "./SubscriptionFeeModal";
import { useSubscriptionFeeActions } from "./useSubscriptionFeeActions";

interface SubscriptionCostsProps extends SubscriptionUsageOptions {
  meteredSpend: number;
  canEdit: boolean;
}

const SubscriptionCosts: React.FC<SubscriptionCostsProps> = ({ meteredSpend, canEdit, ...usageOptions }) => {
  const usage = useSubscriptionUsage(usageOptions);
  const feeActions = useSubscriptionFeeActions({ accessToken: usageOptions.accessToken, refetch: usage.refetch });
  return (
    <>
      <SubscriptionCostCard
        loading={usage.loading}
        failed={usage.failed}
        isDateChanging={false}
        usage={usage.data}
        meteredSpend={meteredSpend}
        canEdit={canEdit}
        onSetFee={feeActions.openFeeModal}
        onEditFee={feeActions.openFeeModal}
        onRemoveFee={feeActions.removeFee}
      />
      <SubscriptionFeeModal
        visible={feeActions.feeModalAccount !== null}
        account={feeActions.feeModalAccount}
        onCancel={feeActions.closeFeeModal}
        onSubmit={feeActions.submitFee}
      />
    </>
  );
};

export default SubscriptionCosts;
