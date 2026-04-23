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
  useCreateAccessGroup,
  AccessGroupCreateParams,
} from "@/app/(dashboard)/hooks/accessGroups/useCreateAccessGroup";
import { FormProvider, useForm } from "react-hook-form";
import { useEffect } from "react";

interface AccessGroupCreateModalProps {
  visible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

const emptyValues: AccessGroupFormValues = {
  name: "",
  description: "",
  modelIds: [],
  mcpServerIds: [],
  agentIds: [],
};

export function AccessGroupCreateModal({
  visible,
  onCancel,
  onSuccess,
}: AccessGroupCreateModalProps) {
  const form = useForm<AccessGroupFormValues>({
    defaultValues: emptyValues,
    mode: "onSubmit",
  });
  const createMutation = useCreateAccessGroup();

  // Reset whenever the dialog reopens — sonner equivalent of antd
  // destroyOnClose is a deliberate form.reset().
  useEffect(() => {
    if (visible) form.reset(emptyValues);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible]);

  const onSubmit = form.handleSubmit((values) => {
    const params: AccessGroupCreateParams = {
      access_group_name: values.name,
      description: values.description,
      access_model_names: values.modelIds,
      access_mcp_server_ids: values.mcpServerIds,
      access_agent_ids: values.agentIds,
    };
    createMutation.mutate(params, {
      onSuccess: () => {
        MessageManager.success("Access group created successfully");
        form.reset(emptyValues);
        onSuccess?.();
        onCancel();
      },
    });
  });

  return (
    <Dialog open={visible} onOpenChange={(o) => (!o ? onCancel() : undefined)}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Create Access Group</DialogTitle>
          <DialogDescription className="sr-only">
            Create a new access group and assign models, MCP servers, and agents.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit}>
            <AccessGroupBaseForm />
            <DialogFooter className="mt-6">
              <Button type="button" variant="outline" onClick={onCancel}>
                Cancel
              </Button>
              <Button type="submit" disabled={createMutation.isPending}>
                {createMutation.isPending ? "Creating…" : "Create Group"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}
