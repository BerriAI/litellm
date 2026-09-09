"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import EditAdeptRouterModal, { type AdeptRouterModelData } from "./EditAdeptRouterModal";

interface AdeptRouterEditControlProps {
  canEdit: boolean;
  isEditing: boolean;
  modelData: AdeptRouterModelData;
  accessToken: string;
  onUpdated: (updated: AdeptRouterModelData) => void;
}

/** Composes the "Edit ADEPT Router" trigger + modal into one unit so model_info_view.tsx does
 *  not have to know about the ADEPT-specific state, mirroring how EditAutoRouterModal is wired
 *  next to it but without adding 15+ lines of view-model glue to the parent. */
export const AdeptRouterEditControl: React.FC<AdeptRouterEditControlProps> = ({
  canEdit,
  isEditing,
  modelData,
  accessToken,
  onUpdated,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const isAdept = (modelData?.litellm_params as { model?: string } | undefined)?.model?.startsWith("adept/") ?? false;
  if (!isAdept) return null;
  return (
    <>
      {canEdit && !isEditing && (
        <Button onClick={() => setIsOpen(true)} className="flex items-center">
          Edit ADEPT Router
        </Button>
      )}
      <EditAdeptRouterModal
        isVisible={isOpen}
        onCancel={() => setIsOpen(false)}
        onSuccess={onUpdated}
        modelData={modelData}
        accessToken={accessToken}
      />
    </>
  );
};

export default AdeptRouterEditControl;
