"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { useState } from "react";
import { FolderPlus } from "lucide-react";

import { toast } from "@/lib/toast";
import TranslatedMessage from "@/components/shared/TranslatedMessage";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useCreateProject, ProjectCreateParams } from "@/app/(dashboard)/hooks/projects/useCreateProject";
import { ProjectBaseForm } from "./ProjectBaseForm";
import { emptyProjectFormValues, projectFormSchema } from "./projectFormSchema";
import { buildProjectCreateParams } from "./projectFormUtils";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function CreateProjectForm({ onClose }: { onClose: () => void }) {
  const { t } = useLanguage();
  // 校验保存固定原文，由字段展示层翻译，避免切换语言后错误和草稿失步。
  const form = useZodForm(projectFormSchema, { defaultValues: emptyProjectFormValues });
  const createMutation = useCreateProject();
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const handleSubmit = form.handleSubmit((values) => {
    const params: ProjectCreateParams = {
      ...buildProjectCreateParams(values),
      team_id: values.team_id,
    };

    createMutation.mutate(params, {
      onSuccess: () => {
        // 通知在显示时读取语言，避免异步回调和已显示通知保留旧语言。
        toast.success(<TranslatedMessage message="Project created successfully" />);
        form.reset(emptyProjectFormValues);
        onClose();
      },
      onError: (error) => {
        toast.error(extractProxyErrorMessage(error) || <TranslatedMessage message="Failed to create project" />);
      },
    });
  });

  const handleCancel = () => {
    form.reset(emptyProjectFormValues);
    onClose();
  };

  return (
    <form onSubmit={(event) => event.preventDefault()}>
      <ProjectBaseForm form={form} advancedOpen={advancedOpen} onAdvancedOpenChange={setAdvancedOpen} />

      <div className="mt-6 flex justify-end gap-2 border-t border-border pt-4">
        <Button type="button" variant="outline" onClick={handleCancel}>
          {t("Cancel")}
        </Button>
        <Button type="button" onClick={() => void handleSubmit()} disabled={createMutation.isPending}>
          {createMutation.isPending ? <UiLoadingSpinner /> : <FolderPlus />}
          {t("Create Project")}
        </Button>
      </div>
    </form>
  );
}

export function CreateProjectModal({ isOpen, onClose }: CreateProjectModalProps) {
  const { t } = useLanguage();
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[720px]">
        <DialogHeader>
          <DialogTitle className="text-lg">{t("Create New Project")}</DialogTitle>
        </DialogHeader>
        <CreateProjectForm onClose={onClose} />
      </DialogContent>
    </Dialog>
  );
}
