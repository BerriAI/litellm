import React, { useState, useEffect } from "react";
import { Icon, Button, Col, Text, Grid } from "@tremor/react";
import { RefreshIcon } from "@heroicons/react/outline";
import { Trans, useTranslation } from "react-i18next";
import TagInfoView from "./tag_info";
import { modelInfoCall } from "../networking";
import { tagCreateCall, tagListCall, tagDeleteCall } from "../networking";
import { Tag } from "./types";
import TagTable from "./TagTable";
import NotificationsManager from "../molecules/notifications_manager";
import CreateTagModal from "./components/CreateTagModal";

interface ModelInfo {
  model_name: string;
  litellm_params: {
    model: string;
  };
  model_info: {
    id: string;
  };
}

interface TagProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

const TagManagement: React.FC<TagProps> = ({ accessToken, userID, userRole }) => {
  const { t } = useTranslation();
  const [tags, setTags] = useState<Tag[]>([]);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [selectedTagId, setSelectedTagId] = useState<string | null>(null);
  const [editTag, setEditTag] = useState<boolean>(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [tagToDelete, setTagToDelete] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);

  const fetchTags = async () => {
    if (!accessToken) return;
    try {
      const response = await tagListCall(accessToken);
      console.log("List tags response:", response);
      setTags(Object.values(response));
    } catch (error) {
      console.error("Error fetching tags:", error);
      NotificationsManager.fromBackend(t("tagManagement.index.errorFetchingTags", { error }));
    }
  };

  const handleRefreshClick = () => {
    fetchTags();
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  };

  const handleCreate = async (formValues: any) => {
    if (!accessToken) return;
    try {
      await tagCreateCall(accessToken, {
        name: formValues.tag_name,
        description: formValues.description,
        models: formValues.allowed_llms,
        max_budget: formValues.max_budget,
        soft_budget: formValues.soft_budget,
        tpm_limit: formValues.tpm_limit,
        rpm_limit: formValues.rpm_limit,
        budget_duration: formValues.budget_duration,
      });
      NotificationsManager.success(t("tagManagement.index.tagCreatedSuccess"));
      setIsCreateModalVisible(false);
      fetchTags();
    } catch (error) {
      console.error("Error creating tag:", error);
      NotificationsManager.fromBackend(t("tagManagement.index.errorCreatingTag", { error }));
    }
  };

  const handleDelete = async (tagName: string) => {
    setTagToDelete(tagName);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!accessToken || !tagToDelete) return;
    try {
      await tagDeleteCall(accessToken, tagToDelete);
      NotificationsManager.success(t("tagManagement.index.tagDeletedSuccess"));
      fetchTags();
    } catch (error) {
      console.error("Error deleting tag:", error);
      NotificationsManager.fromBackend(t("tagManagement.index.errorDeletingTag", { error }));
    }
    setIsDeleteModalOpen(false);
    setTagToDelete(null);
  };

  useEffect(() => {
    if (userID && userRole && accessToken) {
      const fetchModels = async () => {
        try {
          const response = await modelInfoCall(accessToken, userID, userRole);
          if (response && response.data) {
            setAvailableModels(response.data);
          }
        } catch (error) {
          console.error("Error fetching models:", error);
          NotificationsManager.fromBackend(t("tagManagement.index.errorFetchingModels", { error }));
        }
      };
      fetchModels();
    }
  }, [accessToken, userID, userRole]);

  useEffect(() => {
    fetchTags();
  }, [accessToken]);

  return (
    <div className="w-full mx-4 h-[75vh]">
      {selectedTagId ? (
        <TagInfoView
          tagId={selectedTagId}
          onClose={() => {
            setSelectedTagId(null);
            setEditTag(false);
          }}
          accessToken={accessToken}
          is_admin={userRole === "Admin"}
          editTag={editTag}
        />
      ) : (
        <div className="gap-2 p-8 h-[75vh] w-full mt-2">
          <div className="flex justify-between mt-2 w-full items-center mb-4">
            <h1>{t("tagManagement.index.pageTitle")}</h1>
            <div className="flex items-center space-x-2">
              {lastRefreshed && <Text>{t("tagManagement.index.lastRefreshed", { time: lastRefreshed })}</Text>}
              <Icon
                icon={RefreshIcon}
                variant="shadow"
                size="xs"
                className="self-center cursor-pointer"
                onClick={handleRefreshClick}
              />
            </div>
          </div>

          <Text className="mb-4">
            {t("tagManagement.index.clickTagNameHint")}
            <p>
              <Trans
                i18nKey="tagManagement.index.tagRoutingDescription"
                components={{
                  docsLink: (
                    <a
                      href="https://docs.litellm.ai/docs/proxy/tag_routing"
                      target="_blank"
                      rel="noopener noreferrer"
                    />
                  ),
                }}
              />
            </p>
          </Text>

          <Button className="mb-4" onClick={() => setIsCreateModalVisible(true)}>
            {t("tagManagement.index.createNewTagButton")}
          </Button>

          <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
            <Col numColSpan={1}>
              <TagTable
                data={tags}
                onEdit={(tag) => {
                  setSelectedTagId(tag.name);
                  setEditTag(true);
                }}
                onDelete={handleDelete}
                onSelectTag={setSelectedTagId}
              />
            </Col>
          </Grid>

          {/* Create Tag Modal */}
          <CreateTagModal
            visible={isCreateModalVisible}
            onCancel={() => setIsCreateModalVisible(false)}
            onSubmit={handleCreate}
            availableModels={availableModels}
          />

          {/* Delete Confirmation Modal */}
          {isDeleteModalOpen && (
            <div className="fixed z-10 inset-0 overflow-y-auto">
              <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                <div className="fixed inset-0 transition-opacity" aria-hidden="true">
                  <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
                </div>
                <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                  <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                    <div className="sm:flex sm:items-start">
                      <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                        <h3 className="text-lg leading-6 font-medium text-gray-900">
                          {t("tagManagement.index.deleteTagTitle")}
                        </h3>
                        <div className="mt-2">
                          <p className="text-sm text-gray-500">{t("tagManagement.index.deleteTagConfirm")}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                    <Button onClick={confirmDelete} color="red" className="ml-2">
                      {t("common.delete")}
                    </Button>
                    <Button
                      onClick={() => {
                        setIsDeleteModalOpen(false);
                        setTagToDelete(null);
                      }}
                    >
                      {t("common.cancel")}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TagManagement;
