"use client";

import PoliciesPanel from "@/components/policies";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  getPoliciesList,
  createPolicyCall,
  updatePolicyCall,
  deletePolicyCall,
  getPolicyInfo,
  getPolicyAttachmentsList,
  createPolicyAttachmentCall,
  deletePolicyAttachmentCall,
  getGuardrailsList,
} from "@/components/networking";

const PoliciesPage = () => {
  const { accessToken, userRole } = useAuthorized();

  return (
    <PoliciesPanel
      accessToken={accessToken}
      userRole={userRole}
      getPoliciesList={getPoliciesList}
      createPolicy={createPolicyCall}
      updatePolicy={updatePolicyCall}
      deletePolicy={deletePolicyCall}
      getPolicy={getPolicyInfo}
      getAttachmentsList={getPolicyAttachmentsList}
      createAttachment={createPolicyAttachmentCall}
      deleteAttachment={deletePolicyAttachmentCall}
      getGuardrailsList={getGuardrailsList}
    />
  );
};

export default PoliciesPage;
