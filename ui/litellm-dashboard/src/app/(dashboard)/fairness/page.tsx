"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

import { FairnessSettingsForm } from "./_components/FairnessSettingsForm";
import { FairnessStatusTable } from "./_components/FairnessStatusTable";

export default function FairnessPage() {
  const { isViewOnly } = useAuthorized();
  return (
    <div className="flex flex-col gap-6 p-6">
      <FairnessSettingsForm readOnly={isViewOnly} />
      <FairnessStatusTable />
    </div>
  );
}
