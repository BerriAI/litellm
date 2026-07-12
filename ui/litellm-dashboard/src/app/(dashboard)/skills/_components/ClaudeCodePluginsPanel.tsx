import React, { useState, useEffect } from "react";
import { Button } from "@tremor/react";
import { Modal } from "antd";
import {
  getClaudeCodePluginsList,
  deleteClaudeCodePlugin,
  getClaudeCodeMarketplaces,
  syncClaudeCodeMarketplace,
  deleteClaudeCodeMarketplace,
} from "@/components/networking";
import AddPluginForm from "./add_plugin_form";
import AddMarketplaceForm from "./add_marketplace_form";
import MarketplaceTable from "./marketplace_table";
import PluginTable from "./plugin_table";
import SkillDetail from "@/components/claude_code_plugins/skill_detail";
import { isAdminRole } from "@/utils/roles";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  Plugin,
  ListPluginsResponse,
  MarketplaceSource,
  ListMarketplacesResponse,
} from "@/components/claude_code_plugins/types";

interface ClaudeCodePluginsPanelProps {
  accessToken: string | null;
  userRole?: string;
}

const ClaudeCodePluginsPanel: React.FC<ClaudeCodePluginsPanelProps> = ({ accessToken, userRole }) => {
  const [pluginsList, setPluginsList] = useState<Plugin[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [pluginToDelete, setPluginToDelete] = useState<{
    name: string;
    displayName: string;
  } | null>(null);
  const [selectedSkill, setSelectedSkill] = useState<Plugin | null>(null);

  const [marketplacesList, setMarketplacesList] = useState<MarketplaceSource[]>([]);
  const [isAddMarketplaceModalVisible, setIsAddMarketplaceModalVisible] = useState(false);
  const [isMarketplacesLoading, setIsMarketplacesLoading] = useState(false);
  const [isDeletingMarketplace, setIsDeletingMarketplace] = useState(false);
  const [syncingMarketplaceName, setSyncingMarketplaceName] = useState<string | null>(null);
  const [marketplaceToDelete, setMarketplaceToDelete] = useState<{
    name: string;
    displayName: string;
  } | null>(null);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchPlugins = async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const response: ListPluginsResponse = await getClaudeCodePluginsList(accessToken, false);
      setPluginsList(response.plugins);
    } catch (error) {
      console.error("Error fetching skills:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchMarketplaces = async () => {
    if (!accessToken) return;

    setIsMarketplacesLoading(true);
    try {
      const response: ListMarketplacesResponse = await getClaudeCodeMarketplaces(accessToken);
      setMarketplacesList(response.marketplaces);
    } catch (error) {
      console.error("Error fetching marketplaces:", error);
    } finally {
      setIsMarketplacesLoading(false);
    }
  };

  useEffect(() => {
    fetchPlugins();
    fetchMarketplaces();
  }, [accessToken]);

  const handleMarketplaceAdded = () => {
    fetchMarketplaces();
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
      NotificationsManager.success(`Skill "${pluginToDelete.displayName}" deleted successfully`);
      fetchPlugins();
    } catch (error) {
      console.error("Error deleting skill:", error);
      NotificationsManager.error("Failed to delete skill");
    } finally {
      setIsDeleting(false);
      setPluginToDelete(null);
    }
  };

  const handleSyncMarketplace = async (marketplaceName: string) => {
    if (!accessToken) return;

    setSyncingMarketplaceName(marketplaceName);
    try {
      await syncClaudeCodeMarketplace(accessToken, marketplaceName);
      NotificationsManager.success(`Marketplace "${marketplaceName}" synced successfully`);
      fetchMarketplaces();
      fetchPlugins();
    } catch (error) {
      console.error("Error syncing marketplace:", error);
      NotificationsManager.error("Failed to sync marketplace");
    } finally {
      setSyncingMarketplaceName(null);
    }
  };

  const handleDeleteMarketplaceClick = (marketplaceName: string, displayName: string) => {
    setMarketplaceToDelete({ name: marketplaceName, displayName });
  };

  const handleDeleteMarketplaceConfirm = async () => {
    if (!marketplaceToDelete || !accessToken) return;

    setIsDeletingMarketplace(true);
    try {
      await deleteClaudeCodeMarketplace(accessToken, marketplaceToDelete.name);
      NotificationsManager.success(`Marketplace "${marketplaceToDelete.displayName}" deleted successfully`);
      fetchMarketplaces();
      fetchPlugins();
    } catch (error) {
      console.error("Error deleting marketplace:", error);
      NotificationsManager.error("Failed to delete marketplace");
    } finally {
      setIsDeletingMarketplace(false);
      setMarketplaceToDelete(null);
    }
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      {selectedSkill ? (
        <SkillDetail
          skill={selectedSkill}
          onBack={() => setSelectedSkill(null)}
          isAdmin={isAdmin}
          accessToken={accessToken}
          onPublishClick={fetchPlugins}
        />
      ) : (
        <>
          <div className="flex flex-col gap-2 mb-4">
            <h1 className="text-2xl font-bold">Marketplaces</h1>
            <p className="text-sm text-gray-600">
              Import external Claude Code plugin marketplaces. Their skills are namespaced (e.g.{" "}
              <code className="bg-gray-100 px-1 rounded-sm">marketplace-name--skill-name</code>) and can be granted to
              orgs, teams, or keys.
            </p>
            <div className="mt-2 flex gap-2">
              <Button onClick={() => setIsAddMarketplaceModalVisible(true)} disabled={!accessToken || !isAdmin}>
                + Add Marketplace
              </Button>
            </div>
          </div>

          <MarketplaceTable
            marketplacesList={marketplacesList}
            isLoading={isMarketplacesLoading}
            isAdmin={isAdmin}
            syncingName={syncingMarketplaceName}
            onSyncClick={handleSyncMarketplace}
            onDeleteClick={handleDeleteMarketplaceClick}
          />

          <div className="flex flex-col gap-2 mb-4 mt-10">
            <h1 className="text-2xl font-bold">Skills</h1>
            <p className="text-sm text-gray-600">
              Register Claude Code skills. Public skills appear in the Skill Hub for all users and are served via{" "}
              <code className="bg-gray-100 px-1 rounded-sm">/claude-code/marketplace.json</code>. Non-public skills
              require a per-skill grant on an org, team, or key.
            </p>
            <div className="mt-2 flex gap-2">
              <Button onClick={() => setIsAddModalVisible(true)} disabled={!accessToken || !isAdmin}>
                + Add Skill
              </Button>
            </div>
          </div>

          <PluginTable
            pluginsList={pluginsList}
            isLoading={isLoading}
            onDeleteClick={handleDeleteClick}
            accessToken={accessToken}
            isAdmin={isAdmin}
            onPluginClick={(id) => {
              const skill = pluginsList.find((p) => p.id === id);
              if (skill) setSelectedSkill(skill);
            }}
          />
        </>
      )}

      <AddPluginForm
        visible={isAddModalVisible}
        onClose={() => setIsAddModalVisible(false)}
        accessToken={accessToken}
        onSuccess={fetchPlugins}
      />

      <AddMarketplaceForm
        visible={isAddMarketplaceModalVisible}
        onClose={() => setIsAddMarketplaceModalVisible(false)}
        accessToken={accessToken}
        onSuccess={handleMarketplaceAdded}
      />

      {pluginToDelete && (
        <Modal
          title="Delete Skill"
          open={pluginToDelete !== null}
          onOk={handleDeleteConfirm}
          onCancel={() => setPluginToDelete(null)}
          confirmLoading={isDeleting}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <p>
            Are you sure you want to delete skill: <strong>{pluginToDelete.displayName}</strong>?
          </p>
          <p>This action cannot be undone.</p>
        </Modal>
      )}

      {marketplaceToDelete && (
        <Modal
          title="Delete Marketplace"
          open={marketplaceToDelete !== null}
          onOk={handleDeleteMarketplaceConfirm}
          onCancel={() => setMarketplaceToDelete(null)}
          confirmLoading={isDeletingMarketplace}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <p>
            Are you sure you want to delete marketplace: <strong>{marketplaceToDelete.displayName}</strong>?
          </p>
          <p>This does not delete skills already imported from it, but they will no longer sync.</p>
        </Modal>
      )}
    </div>
  );
};

export default ClaudeCodePluginsPanel;
