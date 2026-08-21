"use client";

import { useState } from "react";
import { Save } from "lucide-react";

import { toast } from "@/lib/toast";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";
import { useUpdateProject, ProjectUpdateParams } from "@/app/(dashboard)/hooks/projects/useUpdateProject";
import { ProjectBaseForm } from "./ProjectBaseForm";
import { projectFormSchema, type ProjectFormValues } from "./projectFormSchema";
import { buildProjectUpdateParams } from "./projectFormUtils";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface EditProjectModalProps {
  isOpen: boolean;
  project: ProjectResponse;
  onClose: () => void;
  onSuccess?: () => void;
}

const INTERNAL_METADATA_KEYS = new Set([
  "model_rpm_limit",
  "model_tpm_limit",
  "model_itpm_limit",
  "model_otpm_limit",
  "guardrails",
]);

export const toFormValues = (project: ProjectResponse): ProjectFormValues => {
  const metadataObj = (project.metadata ?? {}) as Record<string, unknown>;
  const rpmLimits = (metadataObj.model_rpm_limit ?? {}) as Record<string, number>;
  const tpmLimits = (metadataObj.model_tpm_limit ?? {}) as Record<string, number>;
  const itpmLimits = (metadataObj.model_itpm_limit ?? {}) as Record<string, number>;
  const otpmLimits = (metadataObj.model_otpm_limit ?? {}) as Record<string, number>;
  const guardrails = (Array.isArray(metadataObj.guardrails) ? metadataObj.guardrails : []) as string[];

  const modelLimits = Array.from(
    new Set([
      ...Object.keys(rpmLimits),
      ...Object.keys(tpmLimits),
      ...Object.keys(itpmLimits),
      ...Object.keys(otpmLimits),
    ]),
  ).map((model) => ({
    model,
    rpm: rpmLimits[model],
    tpm: tpmLimits[model],
    itpm: itpmLimits[model],
    otpm: otpmLimits[model],
  }));

  const metadata = Object.entries(metadataObj)
    .filter(([key]) => !INTERNAL_METADATA_KEYS.has(key))
    .map(([key, value]) => ({ key, value: String(value) }));

  return {
    project_alias: project.project_alias ?? "",
    team_id: project.team_id ?? "",
    description: project.description ?? "",
    models: project.models ?? [],
    max_budget: project.litellm_budget_table?.max_budget ?? undefined,
    isBlocked: project.blocked,
    guardrails: guardrails.length > 0 ? guardrails : undefined,
    modelLimits: modelLimits.length > 0 ? modelLimits : undefined,
    metadata: metadata.length > 0 ? metadata : undefined,
  };
};

function EditProjectForm({ project, onClose, onSuccess }: Omit<EditProjectModalProps, "isOpen">) {
  const form = useZodForm(projectFormSchema, { defaultValues: toFormValues(project) });
  const updateMutation = useUpdateProject();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [advancedEverOpened, setAdvancedEverOpened] = useState(false);

  const handleAdvancedOpenChange = (open: boolean) => {
    setAdvancedOpen(open);
    if (open) setAdvancedEverOpened(true);
  };

  const handleSubmit = form.handleSubmit((values) => {
    const submitted: ProjectFormValues = advancedEverOpened
      ? values
      : { ...values, guardrails: undefined, modelLimits: undefined, metadata: undefined };

    const params: ProjectUpdateParams = {
      ...buildProjectUpdateParams(submitted),
      team_id: submitted.team_id,
    };

    updateMutation.mutate(
      { projectId: project.project_id, params },
      {
        onSuccess: () => {
          toast.success("Project updated successfully");
          onSuccess?.();
          onClose();
        },
        onError: (error) => {
          toast.error(error.message || "Failed to update project");
        },
      },
    );
  });

  return (
    <form onSubmit={(event) => event.preventDefault()}>
      <ProjectBaseForm form={form} advancedOpen={advancedOpen} onAdvancedOpenChange={handleAdvancedOpenChange} />

      <div className="mt-6 flex justify-end gap-2 border-t border-border pt-4">
        <Button type="button" variant="outline" onClick={onClose}>
          Cancel
        </Button>
        <Button type="button" onClick={() => void handleSubmit()} disabled={updateMutation.isPending}>
          {updateMutation.isPending ? <UiLoadingSpinner /> : <Save />}
          Save Changes
        </Button>
      </div>
    </form>
  );
}

export function EditProjectModal({ isOpen, project, onClose, onSuccess }: EditProjectModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[720px]">
        <DialogHeader>
          <DialogTitle className="text-lg">Edit Project</DialogTitle>
        </DialogHeader>
        <EditProjectForm key={project.project_id} project={project} onClose={onClose} onSuccess={onSuccess} />
      </DialogContent>
    </Dialog>
  );
}
