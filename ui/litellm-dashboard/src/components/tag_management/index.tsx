import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { RefreshCw } from "lucide-react";
import TagInfoView from "./tag_info";
import { modelInfoCall } from "../networking";
import { tagCreateCall, tagListCall, tagDeleteCall } from "../networking";
import { Tag } from "./types";
import TagTable from "./TagTable";
import NotificationsManager from "../molecules/notifications_manager";
import CreateTagModal from "./components/CreateTagModal";

interface ModelInfo {
  model_name: string;
  litellm_params: { model: string };
  model_info: { id: string };
}

interface TagProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

const TagManagement: React.FC<TagProps> = ({
  accessToken,
  userID,
  userRole,
}) => {
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
      setTags(Object.values(response));
    } catch (error) {
      console.error("Error fetching tags:", error);
      NotificationsManager.fromBackend("Error fetching tags: " + error);
    }
  };

  const handleRefreshClick = () => {
    fetchTags();
    setLastRefreshed(new Date().toLocaleString());
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
      NotificationsManager.success("Tag created successfully");
      setIsCreateModalVisible(false);
      fetchTags();
    } catch (error) {
      console.error("Error creating tag:", error);
      NotificationsManager.fromBackend("Error creating tag: " + error);
    }
  };

  const handleDelete = (tagName: string) => {
    setTagToDelete(tagName);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!accessToken || !tagToDelete) return;
    try {
      await tagDeleteCall(accessToken, tagToDelete);
      NotificationsManager.success("Tag deleted successfully");
      fetchTags();
    } catch (error) {
      console.error("Error deleting tag:", error);
      NotificationsManager.fromBackend("Error deleting tag: " + error);
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
          NotificationsManager.fromBackend("Error fetching models: " + error);
        }
      };
      fetchModels();
    }
  }, [accessToken, userID, userRole]);

  useEffect(() => {
    fetchTags();
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
            <h1 className="text-xl font-semibold">Tag Management</h1>
            <div className="flex items-center space-x-2">
              {lastRefreshed && (
                <span className="text-sm text-muted-foreground">
                  Last Refreshed: {lastRefreshed}
                </span>
              )}
              <Button
                variant="outline"
                size="icon"
                onClick={handleRefreshClick}
                aria-label="Refresh"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mb-4">
            Click on a tag name to view and edit its details.
            <br />
            You can use tags to restrict the usage of certain LLMs based on
            tags passed in the request. Read more about tag routing{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/tag_routing"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              here
            </a>
            .
          </p>

          <Button
            className="mb-4"
            onClick={() => setIsCreateModalVisible(true)}
          >
            + Create New Tag
          </Button>

          <div className="pt-2 pb-2 h-[75vh] w-full mt-2">
            <TagTable
              data={tags}
              onEdit={(tag) => {
                setSelectedTagId(tag.name);
                setEditTag(true);
              }}
              onDelete={handleDelete}
              onSelectTag={setSelectedTagId}
            />
          </div>

          <CreateTagModal
            visible={isCreateModalVisible}
            onCancel={() => setIsCreateModalVisible(false)}
            onSubmit={handleCreate}
            availableModels={availableModels}
          />

          <Dialog
            open={isDeleteModalOpen}
            onOpenChange={(o) => {
              if (!o) {
                setIsDeleteModalOpen(false);
                setTagToDelete(null);
              }
            }}
          >
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle>Delete Tag</DialogTitle>
                <DialogDescription>
                  Are you sure you want to delete this tag?
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => {
                    setIsDeleteModalOpen(false);
                    setTagToDelete(null);
                  }}
                >
                  Cancel
                </Button>
                <Button variant="destructive" onClick={confirmDelete}>
                  Delete
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      )}
    </div>
  );
};

export default TagManagement;
