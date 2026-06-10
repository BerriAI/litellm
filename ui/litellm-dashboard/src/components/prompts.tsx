import React, { useState, useEffect } from "react";

import { Button } from "@tremor/react";
import { Modal, Select } from "antd";
import { getPromptsList, PromptSpec, ListPromptsResponse, deletePromptCall } from "./networking";
import PromptTable from "./prompts/prompt_table";
import PromptInfoView from "./prompts/prompt_info";
import AddPromptForm from "./prompts/add_prompt_form";
import PromptEditorView from "./prompts/prompt_editor_view";
import NotificationsManager from "./molecules/notifications_manager";
import { isAdminRole, isProxyAdminRole } from "@/utils/roles";
import { useTranslation } from "react-i18next";

interface PromptsProps {
  accessToken: string | null;
  userRole?: string;
}

const PromptsPanel: React.FC<PromptsProps> = ({ accessToken, userRole }) => {
  const { t } = useTranslation();
  const [promptsList, setPromptsList] = useState<PromptSpec[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedEnvironment, setSelectedEnvironment] = useState<string | undefined>(undefined);
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [showEditorView, setShowEditorView] = useState(false);
  const [editPromptData, setEditPromptData] = useState<any>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [promptToDelete, setPromptToDelete] = useState<{ id: string; name: string } | null>(null);

  const isAdmin = userRole ? isAdminRole(userRole) : false;
  // Admin Viewer follows the read-parity rule: see prompts, no writes.
  const canModify = userRole ? isProxyAdminRole(userRole) : false;

  const fetchPrompts = async () => {
    if (!accessToken) {
      return;
    }

    setIsLoading(true);
    try {
      const response: ListPromptsResponse = await getPromptsList(accessToken, selectedEnvironment);
      console.log(`prompts: ${JSON.stringify(response)}`);
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

  const handlePromptClick = (promptId: string) => {
    setSelectedPromptId(promptId);
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

  const handleDeleteClick = (promptId: string, promptName: string) => {
    setPromptToDelete({ id: promptId, name: promptName });
  };

  const handleDeleteConfirm = async () => {
    if (!promptToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deletePromptCall(accessToken, promptToDelete.id);
      NotificationsManager.success(t("prompts.deleteSuccess", { name: promptToDelete.name }));
      fetchPrompts(); // Refresh the list
    } catch (error) {
      console.error("Error deleting prompt:", error);
      NotificationsManager.fromBackend(t("prompts.deleteFailed"));
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
                    {t("prompts.addNewPrompt")}
                  </Button>
                  <Button onClick={handleAddPromptFromFile} disabled={!accessToken} variant="secondary">
                    {t("prompts.uploadPromptFile")}
                  </Button>
                </>
              )}
            </div>
            <Select
              placeholder={t("prompts.allEnvironments")}
              allowClear
              value={selectedEnvironment}
              onChange={(value) => setSelectedEnvironment(value)}
              style={{ width: 180 }}
              options={[
                { label: t("prompts.envDevelopment"), value: "development" },
                { label: t("prompts.envStaging"), value: "staging" },
                { label: t("prompts.envProduction"), value: "production" },
              ]}
            />
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
        <Modal
          title={t("prompts.deleteModalTitle")}
          open={promptToDelete !== null}
          onOk={handleDeleteConfirm}
          onCancel={handleDeleteCancel}
          confirmLoading={isDeleting}
          okText={t("common.delete")}
          okButtonProps={{ danger: true }}
        >
          <p>{t("prompts.deleteConfirm", { name: promptToDelete.name })}</p>
          <p>{t("prompts.deleteCannotUndo")}</p>
        </Modal>
      )}
    </div>
  );
};

export default PromptsPanel;
