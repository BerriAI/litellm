import React, { useState, useEffect } from "react";
import { Button } from "@tremor/react";
import { Modal } from "antd";
import {
  getClaudeCodePluginsList,
  deleteClaudeCodePlugin,
} from "./networking";
import AddPluginForm from "./claude_code_plugins/add_plugin_form";
import PluginTable from "./claude_code_plugins/plugin_table";
import { isAdminRole } from "@/utils/roles";
import PluginInfoView from "./claude_code_plugins/plugin_info";
import NotificationsManager from "./molecules/notifications_manager";
import { Plugin, ListPluginsResponse } from "./claude_code_plugins/types";

interface ClaudeCodePluginsPanelProps {
  accessToken: string | null;
  userRole?: string;
}

const ClaudeCodePluginsPanel: React.FC<ClaudeCodePluginsPanelProps> = ({
  accessToken,
  userRole,
}) => {
  const [pluginsList, setPluginsList] = useState<Plugin[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [pluginToDelete, setPluginToDelete] = useState<{
    name: string;
    displayName: string;
  } | null>(null);
  const [selectedPluginId, setSelectedPluginId] = useState<string | null>(
    null
  );

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchPlugins = async () => {
    if (!accessToken) {
      return;
    }

    setIsLoading(true);
    try {
      const response: ListPluginsResponse = await getClaudeCodePluginsList(
        accessToken,
        false // Get all plugins (enabled and disabled)
      );
      console.log(`Claude Code plugins: ${JSON.stringify(response)}`);
      setPluginsList(response.plugins);
    } catch (error) {
      console.error("Error fetching Claude Code plugins:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPlugins();
  }, [accessToken]);

  const handleAddPlugin = () => {
    if (selectedPluginId) {
      setSelectedPluginId(null);
    }
    setIsAddModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleSuccess = () => {
    fetchPlugins();
  };

  const handleDeleteClick = (pluginName: string, displayName: string) => {
    setPluginToDelete({ name: pluginName, displayName });
  };

  const handleDeleteConfirm = async () => {
    if (!pluginToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deleteClaudeCodePlugin(accessToken, pluginToDelete.name);
      NotificationsManager.success(
        `Plugin "${pluginToDelete.displayName}" deleted successfully`
      );
      fetchPlugins();
    } catch (error) {
      console.error("Error deleting plugin:", error);
      NotificationsManager.error("Failed to delete plugin");
    } finally {
      setIsDeleting(false);
      setPluginToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setPluginToDelete(null);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex flex-col gap-2 mb-4">
        <h1 className="text-2xl font-bold">Claude Code Plugins</h1>
        <p className="text-sm text-gray-600">
          Manage Claude Code marketplace plugins. Add, enable, disable, or
          delete plugins that will be available in your marketplace catalog.
          Enabled plugins will appear in the public marketplace at{" "}
          <code className="bg-gray-100 px-1 rounded">/claude-code/marketplace.json</code>.
        </p>
        <div className="mt-2">
          <Button onClick={handleAddPlugin} disabled={!accessToken || !isAdmin}>
            + Add New Plugin
          </Button>
        </div>
      </div>

      {selectedPluginId ? (
        <PluginInfoView
          pluginId={selectedPluginId}
          onClose={() => setSelectedPluginId(null)}
          accessToken={accessToken}
          isAdmin={isAdmin}
          onPluginUpdated={fetchPlugins}
        />
      ) : (
        <PluginTable
          pluginsList={pluginsList}
          isLoading={isLoading}
          onDeleteClick={handleDeleteClick}
          accessToken={accessToken}
          onPluginUpdated={fetchPlugins}
          isAdmin={isAdmin}
          onPluginClick={(id) => setSelectedPluginId(id)}
        />
      )}

      <AddPluginForm
        visible={isAddModalVisible}
        onClose={handleCloseModal}
        accessToken={accessToken}
        onSuccess={handleSuccess}
      />

      {pluginToDelete && (
        <Modal
          title="Delete Plugin"
          open={pluginToDelete !== null}
          onOk={handleDeleteConfirm}
          onCancel={handleDeleteCancel}
          confirmLoading={isDeleting}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <p>
            Are you sure you want to delete plugin:{" "}
            <strong>{pluginToDelete.displayName}</strong>?
          </p>
          <p>This action cannot be undone.</p>
        </Modal>
      )}
    </div>
  );
};

export default ClaudeCodePluginsPanel;
