import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { deleteGuardrailCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import { formatGuardrailMode, getGuardrailLogoAndName } from "./guardrail_info_helpers";

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
  const { t } = useTranslation();
  const [isDeleting, setIsDeleting] = useState(false);
  const guardrailName = guardrail ? guardrail.guardrail_name || guardrail.guardrail_id : "";
  const providerDisplayName = getGuardrailLogoAndName(guardrail?.litellm_params?.guardrail || "").displayName;

  const handleDelete = async () => {
    if (!guardrail || !accessToken) return;
    setIsDeleting(true);
    try {
      await deleteGuardrailCall(accessToken, guardrail.guardrail_id);
      toast.success(t("guardrails.deleteModal.toast.deleted", { name: guardrailName }));
      onClose();
      onDeleted();
    } catch {
      toast.fromError(t("guardrails.deleteModal.toast.deleteFailed"));
      onClose();
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <DeleteResourceModal
      isOpen={guardrail !== null}
      title={t("guardrails.deleteModal.title")}
      alertMessage={t("guardrails.deleteModal.danglingReferenceWarning")}
      message={t("guardrails.deleteModal.message", { name: guardrailName })}
      resourceInformationTitle={t("guardrails.deleteModal.infoTitle")}
      resourceInformation={[
        { label: t("guardrails.deleteModal.info.name"), value: guardrail?.guardrail_name },
        { label: t("guardrails.deleteModal.info.id"), value: guardrail?.guardrail_id, code: true },
        { label: t("guardrails.deleteModal.info.provider"), value: providerDisplayName },
        { label: t("guardrails.deleteModal.info.mode"), value: formatGuardrailMode(guardrail?.litellm_params?.mode) },
        {
          label: t("guardrails.deleteModal.info.defaultOn"),
          value: guardrail?.litellm_params?.default_on
            ? t("guardrails.deleteModal.info.yes")
            : t("guardrails.deleteModal.info.no"),
        },
      ]}
      onCancel={onClose}
      onOk={handleDelete}
      confirmLoading={isDeleting}
      requiredConfirmation={guardrailName}
    />
  );
}
