import React, { useState, useEffect } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";

import { Plus, Upload } from "lucide-react";
import { getPromptsList, PromptSpec, ListPromptsResponse, deletePromptCall } from "@/components/networking";
import PromptTable from "./PromptTable";
import PromptInfoView from "./prompt_info";
import AddPromptForm from "./add_prompt_form";
import PromptEditorView from "./prompt_editor_view";
import { toast } from "@/lib/toast";
import { isProxyAdminRole } from "@/utils/roles";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const ALL_ENVIRONMENTS_LABEL_KEY: ParseKeys = "prompts.panel.environments.all";

const ENVIRONMENT_OPTIONS: { labelKey: ParseKeys; value: string }[] = [
  { labelKey: "prompts.panel.environments.development", value: "development" },
  { labelKey: "prompts.panel.environments.staging", value: "staging" },
  { labelKey: "prompts.panel.environments.production", value: "production" },
];

// SelectValue falls back to the raw value unless the root can map it to a label.
const ENVIRONMENT_ITEMS = [{ labelKey: ALL_ENVIRONMENTS_LABEL_KEY, value: null }, ...ENVIRONMENT_OPTIONS];

interface PromptsProps {
  accessToken: string | null;
  userRole?: string;
}

const PromptsPanel: React.FC<PromptsProps> = ({ accessToken, userRole }) => {
  const { t } = useTranslation();
  const [promptsList, setPromptsList] = useState<PromptSpec[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedEnvironment, setSelectedEnvironment] = useState<string | undefined>(undefined);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [selectedPromptEnvironment, setSelectedPromptEnvironment] = useState<string | undefined>(undefined);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [showEditorView, setShowEditorView] = useState(false);
  const [editPromptData, setEditPromptData] = useState<any>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [promptToDelete, setPromptToDelete] = useState<{ id: string; name: string; environment: string } | null>(null);

  // Admin Viewer follows the read-parity rule: see prompts, no writes.
  const canModify = userRole ? isProxyAdminRole(userRole) : false;

  const fetchPrompts = async () => {
    if (!accessToken) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    try {
      const response: ListPromptsResponse = await getPromptsList(accessToken, selectedEnvironment);
      setPromptsList(response.prompts);
    } catch (error) {
      console.error("Error fetching prompts:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPrompts();
  }, [accessToken, selectedEnvironment]);

  const handlePromptClick = (promptId: string, environment: string) => {
    setSelectedPromptId(promptId);
    setSelectedPromptEnvironment(environment);
  };

  const handleAddPrompt = () => {
    if (selectedPromptId) {
      setSelectedPromptId(null);
    }
    setEditPromptData(null);
    setShowEditorView(true);
  };

  const handleEditPrompt = (promptData: any) => {
    setEditPromptData(promptData);
    setShowEditorView(true);
  };

  const handleAddPromptFromFile = () => {
    if (selectedPromptId) {
      setSelectedPromptId(null);
    }
    setIsAddModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleCloseEditor = () => {
    setShowEditorView(false);
    setEditPromptData(null);
  };

  const handleSuccess = () => {
    fetchPrompts();
    setShowEditorView(false);
    setEditPromptData(null);
    setSelectedPromptId(null);
  };

  const handleDeleteClick = (promptId: string, promptName: string, environment: string) => {
    setPromptToDelete({ id: promptId, name: promptName, environment });
  };

  const handleDeleteConfirm = async () => {
    if (!promptToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deletePromptCall(accessToken, promptToDelete.id, promptToDelete.environment);
      toast.success(
        t("prompts.panel.toasts.deleted", { name: promptToDelete.name, environment: promptToDelete.environment }),
      );
      fetchPrompts(); // Refresh the list
    } catch (error) {
      console.error("Error deleting prompt:", error);
      toast.fromError(t("prompts.panel.toasts.deleteFailed"));
    } finally {
      setIsDeleting(false);
      setPromptToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setPromptToDelete(null);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      {showEditorView ? (
        <PromptEditorView
          onClose={handleCloseEditor}
          onSuccess={handleSuccess}
          accessToken={accessToken}
          initialPromptData={editPromptData}
        />
      ) : selectedPromptId ? (
        <PromptInfoView
          promptId={selectedPromptId}
          initialEnvironment={selectedPromptEnvironment}
          onClose={() => setSelectedPromptId(null)}
          accessToken={accessToken}
          isAdmin={canModify}
          onDelete={fetchPrompts}
          onEdit={handleEditPrompt}
        />
      ) : (
        <>
          <div className="flex justify-between items-center mb-4">
            <div className="flex gap-2">
              {canModify && (
                <>
                  <Button onClick={handleAddPrompt} disabled={!accessToken}>
                    <Plus />
                    {t("prompts.panel.addNew")}
                  </Button>
                  <Button onClick={handleAddPromptFromFile} disabled={!accessToken} variant="secondary">
                    <Upload />
                    {t("prompts.panel.uploadFile")}
                  </Button>
                </>
              )}
            </div>
            <Select
              items={ENVIRONMENT_ITEMS.map((item) => ({ label: t(item.labelKey), value: item.value }))}
              value={selectedEnvironment ?? null}
              onValueChange={(value) => setSelectedEnvironment((value as string | null) ?? undefined)}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder={t(ALL_ENVIRONMENTS_LABEL_KEY)} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={null}>{t(ALL_ENVIRONMENTS_LABEL_KEY)}</SelectItem>
                {ENVIRONMENT_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {t(option.labelKey)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <PromptTable
            promptsList={promptsList}
            isLoading={isLoading}
            onPromptClick={handlePromptClick}
            onDeleteClick={handleDeleteClick}
            accessToken={accessToken}
            isAdmin={canModify}
          />
        </>
      )}

      <AddPromptForm
        visible={isAddModalVisible}
        onClose={handleCloseModal}
        accessToken={accessToken}
        onSuccess={handleSuccess}
      />

      {promptToDelete && (
        <AlertDialog
          open
          onOpenChange={(open) => {
            if (!open && !isDeleting) handleDeleteCancel();
          }}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t("prompts.panel.deleteDialog.title")}</AlertDialogTitle>
              <AlertDialogDescription>
                {t("prompts.panel.deleteDialog.description", {
                  environment: promptToDelete.environment,
                  name: promptToDelete.name,
                })}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={isDeleting}>{t("prompts.panel.deleteDialog.cancel")}</AlertDialogCancel>
              <Button variant="destructive" onClick={handleDeleteConfirm} disabled={isDeleting}>
                {t("prompts.panel.deleteDialog.confirm")}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};

export default PromptsPanel;
