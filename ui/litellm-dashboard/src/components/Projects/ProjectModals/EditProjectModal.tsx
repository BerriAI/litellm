import { useEffect } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import MessageManager from "@/components/molecules/message_manager";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";
import {
  useUpdateProject,
  ProjectUpdateParams,
} from "@/app/(dashboard)/hooks/projects/useUpdateProject";
import {
  ProjectBaseForm,
  ProjectFormValues,
  emptyProjectFormValues,
} from "./ProjectBaseForm";
import { buildProjectApiParams } from "./projectFormUtils";

interface EditProjectModalProps {
  isOpen: boolean;
  project: ProjectResponse;
  onClose: () => void;
  onSuccess?: () => void;
}

function buildDefaultsFromProject(
  project: ProjectResponse,
): ProjectFormValues {
  const metadataObj = (project.metadata ?? {}) as Record<string, unknown>;
  const rpmLimits = (metadataObj.model_rpm_limit ?? {}) as Record<
    string,
    number
  >;
  const tpmLimits = (metadataObj.model_tpm_limit ?? {}) as Record<
    string,
    number
  >;
  const guardrails = (
    Array.isArray(metadataObj.guardrails) ? metadataObj.guardrails : []
  ) as string[];

  const modelLimits: ProjectFormValues["modelLimits"] = [];
  const allLimitModels = new Set([
    ...Object.keys(rpmLimits),
    ...Object.keys(tpmLimits),
  ]);
  for (const model of allLimitModels) {
    modelLimits.push({
      model,
      rpm: rpmLimits[model],
      tpm: tpmLimits[model],
    });
  }

  const internalKeys = new Set([
    "model_rpm_limit",
    "model_tpm_limit",
    "guardrails",
  ]);
  const metadata: ProjectFormValues["metadata"] = [];
  for (const [key, value] of Object.entries(metadataObj)) {
    if (!internalKeys.has(key)) {
      metadata.push({ key, value: String(value) });
    }
  }

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
}

export function EditProjectModal({
  isOpen,
  project,
  onClose,
  onSuccess,
}: EditProjectModalProps) {
  const form = useForm<ProjectFormValues>({
    defaultValues: emptyProjectFormValues,
    mode: "onSubmit",
  });
  const updateMutation = useUpdateProject();

  useEffect(() => {
    if (isOpen && project) {
      form.reset(buildDefaultsFromProject(project));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, project]);

  const onSubmit = form.handleSubmit((values) => {
    const params: ProjectUpdateParams = {
      ...buildProjectApiParams(values),
      team_id: values.team_id,
    };

    updateMutation.mutate(
      { projectId: project.project_id, params },
      {
        onSuccess: () => {
          MessageManager.success("Project updated successfully");
          onSuccess?.();
          onClose();
        },
        onError: (error) => {
          MessageManager.error(error.message || "Failed to update project");
        },
      },
    );
  });

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(o) => (!o ? onClose() : undefined)}
    >
      <DialogContent className="max-w-[720px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold">
            Edit Project
          </DialogTitle>
          <DialogDescription className="sr-only">
            Edit an existing project.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit}>
            <ProjectBaseForm />
            <DialogFooter className="mt-6">
              <Button type="button" variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button type="submit" disabled={updateMutation.isPending}>
                <Save className="h-4 w-4" />
                {updateMutation.isPending ? "Saving…" : "Save Changes"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}
