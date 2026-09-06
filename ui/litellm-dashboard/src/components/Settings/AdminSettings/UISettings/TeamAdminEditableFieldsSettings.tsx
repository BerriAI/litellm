"use client";

import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";

interface TeamAdminEditableFieldsSettingsProps {
  editableFields: readonly string[];
  supportedFields: readonly string[];
  description?: string;
  isUpdating: boolean;
  onUpdate: (settings: { team_admin_editable_team_fields: string[] }) => void;
}

export default function TeamAdminEditableFieldsSettings({
  editableFields,
  supportedFields,
  description,
  isUpdating,
  onUpdate,
}: TeamAdminEditableFieldsSettingsProps) {
  const toggleField = (field: string, checked: boolean) => {
    const next = checked ? [...editableFields, field] : editableFields.filter((item) => item !== field);
    onUpdate({ team_admin_editable_team_fields: next });
  };

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium text-foreground">Team admin editable fields</p>
          <Badge variant={editableFields.length > 0 ? "secondary" : "outline"}>
            {editableFields.length > 0
              ? `${editableFields.length} field${editableFields.length !== 1 ? "s" : ""} enabled`
              : "Team admins cannot edit team settings"}
          </Badge>
        </div>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>

      {supportedFields.length === 0 ? (
        <p className="text-xs italic text-muted-foreground">
          This proxy version does not support enabling any team settings fields for team admins yet.
        </p>
      ) : (
        <div className="ml-4 space-y-2">
          {supportedFields.map((field) => {
            const checkboxId = `team-admin-editable-${field}`;
            return (
              <label key={field} htmlFor={checkboxId} className="flex cursor-pointer items-center gap-2">
                <Checkbox
                  id={checkboxId}
                  checked={editableFields.includes(field)}
                  disabled={isUpdating}
                  onCheckedChange={(checked) => toggleField(field, checked === true)}
                />
                <span className="text-sm text-foreground">{field}</span>
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}
