import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import MessageManager from "@/components/molecules/message_manager";
import {
  AccessGroupBaseForm,
  AccessGroupFormValues,
} from "./AccessGroupBaseForm";
import {
  useEditAccessGroup,
  AccessGroupUpdateParams,
} from "@/app/(dashboard)/hooks/accessGroups/useEditAccessGroup";
import { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { FormProvider, useForm } from "react-hook-form";
import { useEffect } from "react";

interface AccessGroupEditModalProps {
  visible: boolean;
  accessGroup: AccessGroupResponse;
  onCancel: () => void;
  onSuccess?: () => void;
}

export function AccessGroupEditModal({
  visible,
  accessGroup,
  onCancel,
  onSuccess,
}: AccessGroupEditModalProps) {
  const form = useForm<AccessGroupFormValues>({
    defaultValues: {
      name: "",
      description: "",
      modelIds: [],
      mcpServerIds: [],
      agentIds: [],
    },
    mode: "onSubmit",
  });
  const editMutation = useEditAccessGroup();

  useEffect(() => {
    if (visible && accessGroup) {
      form.reset({
        name: accessGroup.access_group_name,
        description: accessGroup.description ?? "",
        modelIds: accessGroup.access_model_names ?? [],
        mcpServerIds: accessGroup.access_mcp_server_ids ?? [],
        agentIds: accessGroup.access_agent_ids ?? [],
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, accessGroup]);

  const onSubmit = form.handleSubmit((values) => {
    const params: AccessGroupUpdateParams = {
      access_group_name: values.name,
      description: values.description,
      access_model_names: values.modelIds,
      access_mcp_server_ids: values.mcpServerIds,
      access_agent_ids: values.agentIds,
    };
    editMutation.mutate(
      { accessGroupId: accessGroup.access_group_id, params },
      {
        onSuccess: () => {
          MessageManager.success("Access group updated successfully");
          onSuccess?.();
          onCancel();
        },
      },
    );
  });

  return (
    <Dialog open={visible} onOpenChange={(o) => (!o ? onCancel() : undefined)}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Edit Access Group</DialogTitle>
          <DialogDescription className="sr-only">
            Edit the access group and its resource assignments.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit}>
            <AccessGroupBaseForm />
            <DialogFooter className="mt-6">
              <Button type="button" variant="outline" onClick={onCancel}>
                Cancel
              </Button>
              <Button type="submit" disabled={editMutation.isPending}>
                {editMutation.isPending ? "Saving…" : "Save Changes"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}
