import React, { useState } from "react";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { deleteGuardrailCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import { formatGuardrailMode, getGuardrailLogoAndName } from "./guardrail_info_helpers";

const DANGLING_REFERENCE_WARNING =
  "Keys, teams, and policies that list this guardrail will silently stop running it. Policy pipeline steps that use it will block every request with a 400 error, unless the step's on_error is next or allow (or its on_fail, when on_error is not set).";

export interface DeletableGuardrail {
  guardrail_id: string;
  guardrail_name?: string | null;
  litellm_params?: {
    guardrail?: string;
    mode?: unknown;
    default_on?: boolean | null;
  } | null;
}

interface GuardrailDeleteModalProps {
  guardrail: DeletableGuardrail | null;
  accessToken: string | null;
  onClose: () => void;
  onDeleted: () => void;
}

export function GuardrailDeleteModal({ guardrail, accessToken, onClose, onDeleted }: GuardrailDeleteModalProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const guardrailName = guardrail ? guardrail.guardrail_name || guardrail.guardrail_id : "";
  const providerDisplayName = getGuardrailLogoAndName(guardrail?.litellm_params?.guardrail || "").displayName;

  const handleDelete = async () => {
    if (!guardrail || !accessToken) return;
    setIsDeleting(true);
    try {
      await deleteGuardrailCall(accessToken, guardrail.guardrail_id);
      toast.success(`Guardrail "${guardrailName}" deleted successfully`);
      onClose();
      onDeleted();
    } catch {
      toast.fromError("Failed to delete guardrail");
      onClose();
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <DeleteResourceModal
      isOpen={guardrail !== null}
      title="Delete Guardrail"
      alertMessage={DANGLING_REFERENCE_WARNING}
      message={`Are you sure you want to delete guardrail: ${guardrailName}? This action cannot be undone.`}
      resourceInformationTitle="Guardrail Information"
      resourceInformation={[
        { label: "Name", value: guardrail?.guardrail_name },
        { label: "ID", value: guardrail?.guardrail_id, code: true },
        { label: "Provider", value: providerDisplayName },
        { label: "Mode", value: formatGuardrailMode(guardrail?.litellm_params?.mode) },
        { label: "Default On", value: guardrail?.litellm_params?.default_on ? "Yes" : "No" },
      ]}
      onCancel={onClose}
      onOk={handleDelete}
      confirmLoading={isDeleting}
      requiredConfirmation={guardrailName}
    />
  );
}
