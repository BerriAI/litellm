import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Controller, FormProvider, useForm } from "react-hook-form";
import { getSSOSettings, updateSSOSettings } from "./networking";
import NotificationManager from "./molecules/notifications_manager";

interface UIAccessControlFormProps {
  accessToken: string | null;
  onSuccess: () => void;
}

interface FormValues {
  ui_access_mode_type: string;
  restricted_sso_group: string;
  sso_group_jwt_field: string;
}

const defaultValues: FormValues = {
  ui_access_mode_type: "",
  restricted_sso_group: "",
  sso_group_jwt_field: "",
};

const UIAccessControlForm: React.FC<UIAccessControlFormProps> = ({
  accessToken,
  onSuccess,
}) => {
  const form = useForm<FormValues>({ defaultValues, mode: "onSubmit" });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadUIAccessSettings = async () => {
      if (!accessToken) return;
      try {
        const ssoData = await getSSOSettings(accessToken);
        if (ssoData && ssoData.values) {
          const uiAccessMode = ssoData.values.ui_access_mode;
          let formValues: FormValues = { ...defaultValues };

          if (uiAccessMode && typeof uiAccessMode === "object") {
            formValues = {
              ui_access_mode_type: uiAccessMode.type ?? "",
              restricted_sso_group: uiAccessMode.restricted_sso_group ?? "",
              sso_group_jwt_field: uiAccessMode.sso_group_jwt_field ?? "",
            };
          } else if (typeof uiAccessMode === "string") {
            formValues = {
              ui_access_mode_type: uiAccessMode,
              restricted_sso_group:
                ssoData.values.restricted_sso_group ?? "",
              sso_group_jwt_field:
                ssoData.values.team_ids_jwt_field ||
                ssoData.values.sso_group_jwt_field ||
                "",
            };
          }
          form.reset(formValues);
        }
      } catch (error) {
        console.error("Failed to load UI access settings:", error);
      }
    };
    loadUIAccessSettings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const handleUIAccessSubmit = form.handleSubmit(async (values) => {
    if (!accessToken) {
      NotificationManager.fromBackend("No access token available");
      return;
    }
    setLoading(true);
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let apiPayload: any;
      if (values.ui_access_mode_type === "all_authenticated_users") {
        apiPayload = { ui_access_mode: "none" };
      } else {
        apiPayload = {
          ui_access_mode: {
            type: values.ui_access_mode_type,
            restricted_sso_group: values.restricted_sso_group,
            sso_group_jwt_field: values.sso_group_jwt_field,
          },
        };
      }
      await updateSSOSettings(accessToken, apiPayload);
      onSuccess();
    } catch (error) {
      console.error("Failed to save UI access settings:", error);
      NotificationManager.fromBackend("Failed to save UI access settings");
    } finally {
      setLoading(false);
    }
  });

  const accessModeType = form.watch("ui_access_mode_type");

  return (
    <div className="p-4">
      <p className="text-sm text-muted-foreground mb-4">
        Configure who can access the UI interface and how group information is
        extracted from JWT tokens.
      </p>

      <FormProvider {...form}>
        <form onSubmit={handleUIAccessSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="ui_access_mode_type" title="Controls who can access the UI interface">
              UI Access Mode
            </Label>
            <Controller
              control={form.control}
              name="ui_access_mode_type"
              render={({ field }) => (
                <Select
                  value={field.value ?? ""}
                  onValueChange={(v) => field.onChange(v)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select access mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all_authenticated_users">
                      All Authenticated Users
                    </SelectItem>
                    <SelectItem value="restricted_sso_group">
                      Restricted SSO Group
                    </SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
          </div>

          {accessModeType === "restricted_sso_group" && (
            <div className="space-y-2">
              <Label htmlFor="restricted_sso_group">
                Restricted SSO Group{" "}
                <span className="text-destructive">*</span>
              </Label>
              <Input
                id="restricted_sso_group"
                placeholder="ui-access-group"
                {...form.register("restricted_sso_group", {
                  required: "Please enter the restricted SSO group",
                })}
              />
              {form.formState.errors.restricted_sso_group && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.restricted_sso_group.message as string}
                </p>
              )}
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="sso_group_jwt_field" title="JWT field name that contains team/group information. Use dot notation to access nested fields.">
              SSO Group JWT Field
            </Label>
            <Input
              id="sso_group_jwt_field"
              placeholder="groups"
              {...form.register("sso_group_jwt_field")}
            />
          </div>

          <div className="flex justify-end">
            <Button type="submit" disabled={loading}>
              {loading ? "Saving…" : "Update UI Access Control"}
            </Button>
          </div>
        </form>
      </FormProvider>
    </div>
  );
};

export default UIAccessControlForm;
