"use client";

import React, { useState } from "react";

import { toast } from "@/lib/toast";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { useEditAccessGroup, AccessGroupUpdateParams } from "@/app/(dashboard)/hooks/accessGroups/useEditAccessGroup";
import { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";

import {
  AccessGroupBaseForm,
  accessGroupFormSchema,
  AGENTS_TAB,
  GENERAL_TAB,
  MCP_SERVERS_TAB,
  MODELS_TAB,
  type AccessGroupFormValues,
} from "./AccessGroupBaseForm";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface AccessGroupEditModalProps {
  visible: boolean;
  accessGroup: AccessGroupResponse;
  onCancel: () => void;
  onSuccess?: () => void;
}

const toFormValues = (accessGroup: AccessGroupResponse): AccessGroupFormValues => ({
  name: accessGroup.access_group_name,
  description: accessGroup.description ?? "",
  modelIds: accessGroup.access_model_names ?? [],
  mcpServerIds: accessGroup.access_mcp_server_ids ?? [],
  agentIds: accessGroup.access_agent_ids ?? [],
});

function AccessGroupEditForm({ accessGroup, onCancel, onSuccess }: Omit<AccessGroupEditModalProps, "visible">) {
  const form = useZodForm(accessGroupFormSchema, { defaultValues: toFormValues(accessGroup) });
  const editMutation = useEditAccessGroup();
  const [activeTab, setActiveTab] = useState(GENERAL_TAB);
  const [visitedTabs, setVisitedTabs] = useState<ReadonlySet<string>>(new Set([GENERAL_TAB]));

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    setVisitedTabs((previous) => new Set([...previous, tab]));
  };

  const handleOk = form.handleSubmit(
    (values) => {
      const params: AccessGroupUpdateParams = {
        access_group_name: values.name,
        description: values.description,
        access_model_names: visitedTabs.has(MODELS_TAB) ? values.modelIds : undefined,
        access_mcp_server_ids: visitedTabs.has(MCP_SERVERS_TAB) ? values.mcpServerIds : undefined,
        access_agent_ids: visitedTabs.has(AGENTS_TAB) ? values.agentIds : undefined,
      };

      editMutation.mutate(
        { accessGroupId: accessGroup.access_group_id, params },
        {
          onSuccess: () => {
            toast.success("Access group updated successfully");
            onSuccess?.();
            onCancel();
          },
        },
      );
    },
    () => setActiveTab(GENERAL_TAB),
  );

  return (
    <form onSubmit={(event) => event.preventDefault()}>
      <AccessGroupBaseForm form={form} activeTab={activeTab} onTabChange={handleTabChange} />

      <div className="mt-6 flex justify-end gap-2">
        <Button type="button" variant="outline" onClick={onCancel} disabled={editMutation.isPending}>
          Cancel
        </Button>
        <Button type="button" onClick={() => void handleOk()} disabled={editMutation.isPending}>
          Save Changes
        </Button>
      </div>
    </form>
  );
}

export function AccessGroupEditModal({ visible, accessGroup, onCancel, onSuccess }: AccessGroupEditModalProps) {
  return (
    <Dialog open={visible} onOpenChange={(open) => !open && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle>Edit Access Group</DialogTitle>
        </DialogHeader>
        <AccessGroupEditForm
          key={accessGroup.access_group_id}
          accessGroup={accessGroup}
          onCancel={onCancel}
          onSuccess={onSuccess}
        />
      </DialogContent>
    </Dialog>
  );
}
