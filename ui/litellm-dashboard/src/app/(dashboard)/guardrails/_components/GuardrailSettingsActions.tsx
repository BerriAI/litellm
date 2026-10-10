import { Code, Trash2 } from "lucide-react";
import React, { useState } from "react";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { deleteGuardrailCall } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { formatGuardrailMode, getGuardrailLogoAndName } from "./guardrail_info_helpers";

const DANGLING_REFERENCE_WARNING =
  "Keys, teams, or policies that reference this guardrail by name will silently stop running it after deletion. Their requests will not return an error.";

interface GuardrailSettingsActionsProps {
  guardrail: {
    guardrail_id: string;
    guardrail_definition_location?: string;
    guardrail_name?: string | null;
    litellm_params?: { guardrail?: string; mode?: unknown; default_on?: boolean | null } | null;
  };
  isEditing: boolean;
  onEdit: () => void;
  onEditCode: () => void;
  accessToken: string | null;
  onDeleted: () => void;
}

export function GuardrailSettingsActions({
  guardrail,
  isEditing,
  onEdit,
  onEditCode,
  accessToken,
  onDeleted,
}: GuardrailSettingsActionsProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const guardrailName = guardrail.guardrail_name || guardrail.guardrail_id;
  const providerDisplayName = getGuardrailLogoAndName(guardrail.litellm_params?.guardrail || "").displayName;

  const handleDelete = async () => {
    if (!accessToken) return;
    setIsDeleting(true);
    try {
      await deleteGuardrailCall(accessToken, guardrail.guardrail_id);
      toast.success(`Guardrail "${guardrailName}" deleted successfully`);
      setIsModalOpen(false);
      onDeleted();
    } catch {
      toast.fromError("Failed to delete guardrail");
      setIsModalOpen(false);
    } finally {
      setIsDeleting(false);
    }
  };

  if (isEditing || guardrail.guardrail_definition_location === "config") return null;

  return (
    <>
      {guardrail.litellm_params?.guardrail === "custom_code" ? (
        <Button variant="outline" onClick={onEditCode}>
          <Code />
          Edit Code
        </Button>
      ) : (
        <Button variant="outline" onClick={onEdit}>
          Edit Settings
        </Button>
      )}
      <Button variant="destructive" onClick={() => setIsModalOpen(true)}>
        <Trash2 />
        Delete
      </Button>
      <DeleteResourceModal
        isOpen={isModalOpen}
        title="Delete Guardrail"
        alertMessage={DANGLING_REFERENCE_WARNING}
        message={`Are you sure you want to delete guardrail: ${guardrailName}? This action cannot be undone.`}
        resourceInformationTitle="Guardrail Information"
        resourceInformation={[
          { label: "Name", value: guardrail.guardrail_name },
          { label: "ID", value: guardrail.guardrail_id, code: true },
          { label: "Provider", value: providerDisplayName },
          { label: "Mode", value: formatGuardrailMode(guardrail.litellm_params?.mode) },
          { label: "Default On", value: guardrail.litellm_params?.default_on ? "Yes" : "No" },
        ]}
        onCancel={() => setIsModalOpen(false)}
        onOk={handleDelete}
        confirmLoading={isDeleting}
        requiredConfirmation={guardrailName}
      />
    </>
  );
}
