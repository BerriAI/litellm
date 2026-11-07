import { CircleHelp } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { z } from "zod";

import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";

import i18n from "@/i18n/i18n";
import { toast } from "@/lib/toast";
import { getSSOSettings, updateSSOSettings } from "./networking";

interface UIAccessControlFormProps {
  accessToken: string | null;
  onSuccess: () => void;
}

const uiAccessControlSchema = z
  .object({
    ui_access_mode_type: z.string().optional(),
    restricted_sso_group: z.string().optional(),
    sso_group_jwt_field: z.string().optional(),
  })
  .superRefine((values, ctx) => {
    if (values.ui_access_mode_type !== "restricted_sso_group" || values.restricted_sso_group) {
      return;
    }
    ctx.addIssue({
      code: "custom",
      path: ["restricted_sso_group"],
      message: i18n.t("adminPanel.uiAccessControl.errors.restrictedGroupRequired"),
    });
  });

type UIAccessControlFormValues = z.output<typeof uiAccessControlSchema>;

const UI_ACCESS_MODE_OPTIONS = [
  { value: "all_authenticated_users", label: "adminPanel.uiAccessControl.modes.allAuthenticatedUsers" },
  { value: "restricted_sso_group", label: "adminPanel.uiAccessControl.modes.restrictedSsoGroup" },
] as const;

const asRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : null;

const asString = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);

const toFormValues = (ssoData: unknown): UIAccessControlFormValues | null => {
  const values = asRecord(asRecord(ssoData)?.values);
  if (!values) {
    return null;
  }

  const nestedAccessMode = asRecord(values.ui_access_mode);
  if (nestedAccessMode) {
    return {
      ui_access_mode_type: asString(nestedAccessMode.type),
      restricted_sso_group: asString(nestedAccessMode.restricted_sso_group),
      sso_group_jwt_field: asString(nestedAccessMode.sso_group_jwt_field),
    };
  }

  const legacyAccessMode = asString(values.ui_access_mode);
  if (legacyAccessMode !== undefined) {
    return {
      ui_access_mode_type: legacyAccessMode,
      restricted_sso_group: asString(values.restricted_sso_group),
      sso_group_jwt_field: asString(values.team_ids_jwt_field) || asString(values.sso_group_jwt_field),
    };
  }

  return null;
};

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const UIAccessControlForm: React.FC<UIAccessControlFormProps> = ({ accessToken, onSuccess }) => {
  const { t } = useTranslation();
  const form = useZodForm(uiAccessControlSchema, { defaultValues: {} });
  const [loading, setLoading] = useState(false);
  const uiAccessModeType = useWatch({ control: form.control, name: "ui_access_mode_type" });

  useEffect(() => {
    const loadUIAccessSettings = async () => {
      if (accessToken) {
        try {
          const formValues = toFormValues(await getSSOSettings(accessToken));
          if (formValues) {
            form.setValue("ui_access_mode_type", formValues.ui_access_mode_type);
            form.setValue("restricted_sso_group", formValues.restricted_sso_group);
            form.setValue("sso_group_jwt_field", formValues.sso_group_jwt_field);
          }
        } catch (error) {
          console.error("Failed to load UI access settings:", error);
        }
      }
    };

    loadUIAccessSettings();
  }, [accessToken, form]);

  const handleUIAccessSubmit = async (formValues: UIAccessControlFormValues) => {
    if (!accessToken) {
      toast.fromError(t("adminPanel.uiAccessControl.toasts.noAccessToken"));
      return;
    }

    setLoading(true);
    try {
      const apiPayload =
        formValues.ui_access_mode_type === "all_authenticated_users"
          ? { ui_access_mode: "none" }
          : {
              ui_access_mode: {
                type: formValues.ui_access_mode_type,
                restricted_sso_group: formValues.restricted_sso_group,
                sso_group_jwt_field: formValues.sso_group_jwt_field,
              },
            };

      await updateSSOSettings(accessToken, apiPayload);
      onSuccess();
    } catch (error) {
      console.error("Failed to save UI access settings:", error);
      toast.fromError(t("adminPanel.uiAccessControl.toasts.saveFailed"));
    } finally {
      setLoading(false);
    }
  };

  const submitMountedValues = (formValues: UIAccessControlFormValues) =>
    handleUIAccessSubmit(
      formValues.ui_access_mode_type === "restricted_sso_group"
        ? formValues
        : { ...formValues, restricted_sso_group: undefined },
    );

  return (
    <TooltipProvider>
      <div className="p-4">
        <div className="mb-4">
          <p className="text-sm text-muted-foreground">{t("adminPanel.uiAccessControl.description")}</p>
        </div>

        <form onSubmit={form.handleSubmit(submitMountedValues)} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="ui_access_mode_type"
              label={labelWithHint(
                t("adminPanel.uiAccessControl.fields.accessMode"),
                t("adminPanel.uiAccessControl.fields.accessModeHint"),
              )}
            >
              {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                <Select
                  items={UI_ACCESS_MODE_OPTIONS.map((option) => ({ value: option.value, label: t(option.label) }))}
                  value={value ?? null}
                  onValueChange={(selected) => onChange(selected ?? undefined)}
                >
                  <SelectTrigger
                    id={id}
                    className="w-full"
                    aria-invalid={ariaInvalid}
                    aria-describedby={ariaDescribedBy}
                  >
                    <SelectValue placeholder={t("adminPanel.uiAccessControl.fields.accessModePlaceholder")} />
                  </SelectTrigger>
                  <SelectContent>
                    {UI_ACCESS_MODE_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {t(option.label)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>

            {uiAccessModeType === "restricted_sso_group" && (
              <FormField
                control={form.control}
                name="restricted_sso_group"
                label={t("adminPanel.uiAccessControl.fields.restrictedGroup")}
              >
                {({ ref, value, ...field }) => (
                  <Input {...field} ref={ref} value={value ?? ""} placeholder="ui-access-group" />
                )}
              </FormField>
            )}

            <FormField
              control={form.control}
              name="sso_group_jwt_field"
              label={labelWithHint(
                t("adminPanel.uiAccessControl.fields.jwtField"),
                t("adminPanel.uiAccessControl.fields.jwtFieldHint"),
              )}
            >
              {({ ref, value, ...field }) => <Input {...field} ref={ref} value={value ?? ""} placeholder="groups" />}
            </FormField>
          </FieldGroup>

          <div className="mt-4 text-right">
            <Button type="submit" disabled={loading}>
              {loading && <UiLoadingSpinner className="size-4" />}
              {t("adminPanel.uiAccessControl.submit")}
            </Button>
          </div>
        </form>
      </div>
    </TooltipProvider>
  );
};

export default UIAccessControlForm;
