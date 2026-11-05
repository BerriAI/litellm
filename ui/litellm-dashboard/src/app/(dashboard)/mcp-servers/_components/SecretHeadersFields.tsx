import { CircleMinus, Plus } from "lucide-react";
import React, { useEffect, useRef } from "react";
import { useFieldArray, useFormContext } from "react-hook-form";
import { useTranslation } from "react-i18next";

import {
  MountedFormField,
  useMountedName,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { textControl } from "./mcpFieldRules";
import { listControl } from "./mcpFormStore";

const fieldClassName = "rounded-lg border-border focus:border-info focus:ring-ring";

const nonBlankRule =
  (message: string) =>
  (value: unknown): true | string =>
    typeof value === "string" && value.trim() !== "" ? true : message;

const SecretHeadersFields: React.FC<{ isEditing?: boolean }> = ({ isEditing = false }) => {
  const { t } = useTranslation();
  const { control } = useFormContext<MountedFormValues>();
  const { fields, append, remove } = useFieldArray({ control: listControl(control), name: "secret_headers" });
  useMountedName("secret_headers");

  const addedStarterRow = useRef(false);
  useEffect(() => {
    if (isEditing || addedStarterRow.current || fields.length > 0) return;
    addedStarterRow.current = true;
    append({});
  }, [isEditing, fields.length, append]);

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        {t(isEditing ? "mcpServers.secretHeadersFields.editIntro" : "mcpServers.secretHeadersFields.intro")}
      </p>
      {fields.map((item, index) => (
        <div key={item.id} className="flex w-full items-baseline gap-4">
          <MountedFormField
            name={["secret_headers", String(index), "header"]}
            className="flex-1"
            rules={{ validate: { required: nonBlankRule(t("mcpServers.secretHeadersFields.headerNameRequired")) } }}
          >
            {(control) => (
              <Input
                {...textControl(control)}
                placeholder={t("mcpServers.secretHeadersFields.headerNamePlaceholder")}
                className={fieldClassName}
              />
            )}
          </MountedFormField>
          <MountedFormField
            name={["secret_headers", String(index), "value"]}
            className="flex-1"
            rules={{ validate: { required: nonBlankRule(t("mcpServers.secretHeadersFields.headerValueRequired")) } }}
          >
            {(control) => (
              <PasswordInput
                {...textControl(control)}
                placeholder={t("mcpServers.secretHeadersFields.headerValuePlaceholder")}
                autoComplete="new-password"
                groupClassName={fieldClassName}
              />
            )}
          </MountedFormField>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            aria-label={t("mcpServers.secretHeadersFields.remove")}
            disabled={fields.length <= 1 && !isEditing}
            onClick={() => remove(index)}
            className="text-muted-foreground hover:text-destructive"
          >
            <CircleMinus />
          </Button>
        </div>
      ))}
      <Button variant="outline" className="w-full border-dashed" onClick={() => append({})}>
        <Plus />
        {t("mcpServers.secretHeadersFields.add")}
      </Button>
    </div>
  );
};

export default SecretHeadersFields;
