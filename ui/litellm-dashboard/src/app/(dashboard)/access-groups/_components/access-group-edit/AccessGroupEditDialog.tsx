"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as React from "react";

import { accessGroupKeys, type AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { usePickDirty } from "@/lib/forms/pickDirty";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchClient } from "@/lib/http/api";

import { AccessGroupFormFields, GENERAL_TAB } from "../access-group-form/AccessGroupFormFields";
import { accessGroupFormSchema } from "../access-group-form/schema";
import { buildAccessGroupPatchBody, formValuesFromAccessGroup, type AccessGroupPatchBody } from "./mapper";

const defaultPatchAccessGroup = async (
  accessGroupId: string,
  body: AccessGroupPatchBody,
): Promise<AccessGroupResponse | undefined> => {
  const { data } = await fetchClient.PATCH("/v1/access_group/{access_group_id}", {
    params: { path: { access_group_id: accessGroupId } },
    body,
  });
  return data;
};

interface AccessGroupEditDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  accessGroup: AccessGroupResponse;
  patchAccessGroup?: (accessGroupId: string, body: AccessGroupPatchBody) => Promise<AccessGroupResponse | undefined>;
}

export const AccessGroupEditDialog = ({
  open,
  onOpenChange,
  accessGroup,
  patchAccessGroup = defaultPatchAccessGroup,
}: AccessGroupEditDialogProps) => {
  const queryClient = useQueryClient();
  const initialValues = React.useMemo(() => formValuesFromAccessGroup(accessGroup), [accessGroup]);
  const form = useZodForm(accessGroupFormSchema, {
    values: initialValues,
    resetOptions: { keepDirtyValues: true },
  });
  const pickDirty = usePickDirty(form.control);
  const [activeTab, setActiveTab] = React.useState(GENERAL_TAB);

  // reset() inherits resetOptions, so keepDirtyValues must be turned off to actually drop the edits
  const discardEdits = () => form.reset(initialValues, { keepDirtyValues: false });

  const closeAndReset = () => {
    discardEdits();
    setActiveTab(GENERAL_TAB);
    onOpenChange(false);
  };

  const mutation = useMutation({
    mutationFn: (body: AccessGroupPatchBody) => patchAccessGroup(accessGroup.access_group_id, body),
    onSuccess: (updated) => {
      NotificationsManager.success("Access group updated successfully");
      if (updated) {
        queryClient.setQueryData(accessGroupKeys.detail(accessGroup.access_group_id), updated);
      }
      queryClient.invalidateQueries({ queryKey: accessGroupKeys.all });
      closeAndReset();
    },
    onError: (error: unknown) =>
      NotificationsManager.fromBackend(error instanceof Error ? error.message : "Failed to update access group"),
  });

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen && mutation.isPending) return;
    if (!nextOpen) {
      discardEdits();
      setActiveTab(GENERAL_TAB);
    }
    onOpenChange(nextOpen);
  };

  const onSubmit = form.handleSubmit(
    (values) => {
      if (mutation.isPending) return;
      mutation.mutate(buildAccessGroupPatchBody(pickDirty(values)));
    },
    // the only validated field (name) lives on the General Info tab
    () => setActiveTab(GENERAL_TAB),
  );

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit Access Group</DialogTitle>
        </DialogHeader>

        <form onSubmit={onSubmit} noValidate>
          <AccessGroupFormFields control={form.control} activeTab={activeTab} onTabChange={setActiveTab} />

          <DialogFooter className="mt-6">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              disabled={mutation.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={mutation.isPending || !form.formState.isDirty}>
              {mutation.isPending ? "Saving..." : "Save Changes"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
