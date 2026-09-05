"use client";

import React, { useMemo } from "react";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";
import { CreateModelAccessGroupParams } from "@/app/(dashboard)/hooks/modelAccessGroups/useCreateModelAccessGroup";
import { createAccessGroupSchema } from "./createAccessGroupSchema";
import { labelWithHint } from "./LabelWithHint";

interface CreateAccessGroupModalProps {
  existingGroups: readonly string[];
  modelOptions: readonly string[];
  isLoadingModels: boolean;
  isSaving: boolean;
  onCancel: () => void;
  onSubmit: (params: CreateModelAccessGroupParams) => void;
}

const CreateAccessGroupModal: React.FC<CreateAccessGroupModalProps> = ({
  existingGroups,
  modelOptions,
  isLoadingModels,
  isSaving,
  onCancel,
  onSubmit,
}) => {
  const schema = useMemo(() => createAccessGroupSchema(new Set(existingGroups)), [existingGroups]);
  const form = useZodForm(schema, { defaultValues: { access_group: "", model_names: [] } });
  const options = useMemo(() => modelOptions.map((name) => ({ label: name, value: name })), [modelOptions]);

  return (
    <Dialog open onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>Create access group</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          The group is added to every deployment of the models you pick. Keys and teams can then be granted the group by
          name, and it can carry one shared budget.
        </p>
        <form onSubmit={form.handleSubmit(onSubmit)} noValidate>
          <TooltipProvider>
            <FieldGroup className="mt-4">
              <FormField
                control={form.control}
                name="access_group"
                label={labelWithHint("Access Group Name", "Free-text label, for example production-models")}
              >
                {(field) => <Input {...field} placeholder="production-models" autoComplete="off" />}
              </FormField>

              <FormField
                control={form.control}
                name="model_names"
                label={labelWithHint(
                  "Models",
                  "Only models with a database deployment are listed. A model defined in config.yaml joins a group through its access_groups entry in the config file",
                )}
              >
                {({ id, value, onChange }) => (
                  <MultiSelect
                    id={id}
                    options={options}
                    value={value}
                    onValueChange={onChange}
                    loading={isLoadingModels}
                    placeholder="Select models"
                    emptyText="No database models found"
                  />
                )}
              </FormField>
            </FieldGroup>

            <div className="mt-6 flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={onCancel}>
                Cancel
              </Button>
              <Button type="submit" disabled={isSaving}>
                {isSaving ? "Creating..." : "Create Access Group"}
              </Button>
            </div>
          </TooltipProvider>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CreateAccessGroupModal;
