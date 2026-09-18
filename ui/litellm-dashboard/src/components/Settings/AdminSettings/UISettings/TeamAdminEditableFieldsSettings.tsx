"use client";

import { Controller } from "react-hook-form";
import { z } from "zod/v4";

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useUpdateUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  parseSupportedTeamAdminEditableFields,
  parseTeamAdminEditableFields,
  teamAdminFieldLabel,
} from "@/components/team/teamAdminEditAccess";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";

const editableFieldsSchema = z.object({ team_admin_editable_team_fields: z.array(z.string()) });

type SaveEditableFields = ReturnType<typeof useUpdateUISettings>["mutate"];

export default function TeamAdminEditableFieldsSettings() {
  const { accessToken } = useAuthorized();
  const { data, isLoading } = useUISettings();
  const { mutate: saveSettings, isPending } = useUpdateUISettings(accessToken);
  const supportedFields = parseSupportedTeamAdminEditableFields(data?.field_schema);
  const savedFields = parseTeamAdminEditableFields(data?.values);
  const enabledFields = supportedFields.filter((field) => savedFields.includes(field));

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardTitle>Team admin editable fields</CardTitle>
          <Badge variant={enabledFields.length > 0 ? "secondary" : "outline"}>
            {enabledFields.length > 0
              ? `${enabledFields.length} field${enabledFields.length !== 1 ? "s" : ""} enabled`
              : "Team admins cannot edit team settings"}
          </Badge>
        </div>
        <CardDescription>
          {data?.field_schema?.properties?.team_admin_editable_team_fields?.description ??
            "Team settings fields a team admin may change on the teams they administer."}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-16 w-full" />
        ) : (
          <TeamAdminEditableFieldsForm
            key={enabledFields.join(",")}
            enabledFields={enabledFields}
            supportedFields={supportedFields}
            isPending={isPending}
            saveSettings={saveSettings}
          />
        )}
      </CardContent>
    </Card>
  );
}

interface TeamAdminEditableFieldsFormProps {
  enabledFields: readonly string[];
  supportedFields: readonly string[];
  isPending: boolean;
  saveSettings: SaveEditableFields;
}

function TeamAdminEditableFieldsForm({
  enabledFields,
  supportedFields,
  isPending,
  saveSettings,
}: TeamAdminEditableFieldsFormProps) {
  const form = useZodForm(editableFieldsSchema, {
    defaultValues: { team_admin_editable_team_fields: [...enabledFields] },
  });
  const submit = form.handleSubmit((values) =>
    saveSettings(values, {
      onSuccess: () => {
        form.reset(values);
        toast.success("Team admin editable fields updated successfully");
      },
      onError: (error) => {
        toast.fromError(error);
      },
    }),
  );

  if (supportedFields.length === 0) {
    return (
      <p className="text-sm italic text-muted-foreground">
        This proxy version does not support enabling any team settings fields for team admins yet.
      </p>
    );
  }

  return (
    <form onSubmit={(event) => void submit(event)} className="space-y-4">
      <Controller
        control={form.control}
        name="team_admin_editable_team_fields"
        render={({ field }) => (
          <div className="space-y-2">
            {supportedFields.map((name) => {
              const checkboxId = `team-admin-editable-${name}`;
              return (
                <label key={name} htmlFor={checkboxId} className="flex cursor-pointer items-center gap-2">
                  <Checkbox
                    id={checkboxId}
                    checked={field.value.includes(name)}
                    disabled={isPending}
                    onCheckedChange={(checked) =>
                      field.onChange(
                        supportedFields.filter((item) => (item === name ? checked : field.value.includes(item))),
                      )
                    }
                  />
                  <span className="text-sm text-foreground">{teamAdminFieldLabel(name)}</span>
                </label>
              );
            })}
          </div>
        )}
      />
      <div className="flex justify-end">
        <Button type="submit" disabled={isPending || !form.formState.isDirty}>
          {isPending ? "Saving..." : "Save"}
        </Button>
      </div>
    </form>
  );
}
