import { parseAsString, useQueryStates } from "nuqs";
import { useEffect, useState } from "react";

import type { Policy } from "@/components/policies/types";

const POLICY_URL_PARSERS = {
  policy: parseAsString.withOptions({ history: "push" }),
  edit_policy: parseAsString.withOptions({ history: "push" }),
};

interface HandedBackPolicy {
  policy: Policy;
  listWhenHandedBack: readonly Policy[];
}

function resolveEditingPolicy(
  policyId: string | null,
  handedBack: HandedBackPolicy | null,
  policies: readonly Policy[],
): Policy | null {
  if (policyId === null) return null;
  const listed = policies.find((policy) => policy.policy_id === policyId);
  if (handedBack?.policy.policy_id !== policyId) return listed ?? null;
  const listCaughtUp = listed !== undefined && handedBack.listWhenHandedBack !== policies;
  return listCaughtUp ? listed : handedBack.policy;
}

interface PolicyListState {
  policies: readonly Policy[];
  hasFetched: boolean;
  isLoading: boolean;
}

export function usePolicyUrlState({ policies, hasFetched, isLoading }: PolicyListState) {
  const [{ policy: selectedPolicyId, edit_policy: editPolicyId }, setPolicyUrl] = useQueryStates(POLICY_URL_PARSERS);
  const [handedBack, setHandedBack] = useState<HandedBackPolicy | null>(null);

  const editingPolicy = resolveEditingPolicy(editPolicyId, handedBack, policies);
  const isEditingPolicyPending = editPolicyId !== null && editingPolicy === null;
  const isEditTargetMissing = isEditingPolicyPending && hasFetched && !isLoading;

  useEffect(() => {
    if (isEditTargetMissing) void setPolicyUrl({ edit_policy: null }, { history: "replace" });
  }, [isEditTargetMissing, setPolicyUrl]);

  const showPolicyVersion = (policy: Policy) => setHandedBack({ policy, listWhenHandedBack: policies });

  const openPolicyEditor = (policy: Policy) => {
    showPolicyVersion(policy);
    if (policy.policy_id === editPolicyId && selectedPolicyId === null) return;
    void setPolicyUrl({ edit_policy: policy.policy_id, policy: selectedPolicyId === null ? undefined : null });
  };

  const closePolicyEditor = () => {
    if (editPolicyId !== null) void setPolicyUrl({ edit_policy: null });
  };

  const selectPolicy = (policyId: string | null) => {
    if (policyId !== selectedPolicyId) void setPolicyUrl({ policy: policyId });
  };

  return {
    selectedPolicyId,
    selectPolicy,
    editingPolicy,
    isEditingPolicyPending,
    openPolicyEditor,
    showPolicyVersion,
    closePolicyEditor,
  };
}
