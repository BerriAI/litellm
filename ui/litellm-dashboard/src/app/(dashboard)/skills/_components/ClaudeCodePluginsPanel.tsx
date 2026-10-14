import React, { useEffect, useState } from "react";
import { Copy } from "lucide-react";
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
import { getClaudeCodePluginsList, deleteClaudeCodePlugin, getProxyBaseUrl } from "@/components/networking";
import AddPluginForm from "./add_plugin_form";
import PluginTable from "./PluginTable";
import SkillDetail from "@/components/claude_code_plugins/skill_detail";
import { isAdminRole } from "@/utils/roles";
import { toast } from "@/lib/toast";
import { Plugin, ListPluginsResponse } from "@/components/claude_code_plugins/types";

interface ClaudeCodePluginsPanelProps {
  accessToken: string | null;
  userRole?: string;
}

const ClaudeCodePluginsPanel: React.FC<ClaudeCodePluginsPanelProps> = ({ accessToken, userRole }) => {
  const [pluginsList, setPluginsList] = useState<Plugin[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);
  const marketplaceUrl = `${getProxyBaseUrl().replace(/\/+$/, "")}/claude-code/marketplace.json`;
  const [pluginToDelete, setPluginToDelete] = useState<{
    name: string;
    displayName: string;
  } | null>(null);
  const [selectedSkill, setSelectedSkill] = useState<Plugin | null>(null);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchPlugins = async () => {
    if (!accessToken) {
      setIsLoading(false);
      return;
    }

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

  useEffect(() => {
    fetchPlugins();
  }, [accessToken]);

  const marketplaceConfig = JSON.stringify(
    [
      {
        source: "url",
        url: marketplaceUrl,
        credentialKind: "inferenceCredential",
        installationPreference: "available",
      },
    ],
    null,
    2,
  );

  const handleCopyMarketplaceConfig = async () => {
    try {
      await navigator.clipboard.writeText(marketplaceConfig);
      toast.success("Claude Desktop configuration copied");
    } catch {
      toast.error("Could not copy Claude Desktop configuration");
    }
  };

  const handleDeleteClick = (pluginName: string, displayName: string) => {
    setPluginToDelete({ name: pluginName, displayName });
  };

  const handleDeleteConfirm = async () => {
    if (!pluginToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deleteClaudeCodePlugin(accessToken, pluginToDelete.name);
      toast.success(`Skill "${pluginToDelete.displayName}" deleted successfully`);
      fetchPlugins();
    } catch (error) {
      console.error("Error deleting skill:", error);
      toast.error("Failed to delete skill");
    } finally {
      setIsDeleting(false);
      setPluginToDelete(null);
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
            <h1 className="text-2xl font-bold">Skills</h1>
            <p className="text-sm text-muted-foreground">
              Register Claude Code skills. Published skills appear in the Skill Hub for all users and are served via{" "}
              <code className="bg-muted px-1 rounded-sm">/claude-code/marketplace.json</code>.
            </p>
            <section className="mt-2 rounded-lg border p-4">
              <h2 className="text-sm font-semibold">Connect Claude Desktop</h2>
              <p className="mt-1 text-sm text-muted-foreground">Marketplace URL</p>
              <code className="break-all text-xs">{marketplaceUrl}</code>
              <div className="mt-3 flex items-start justify-between gap-3">
                <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">{marketplaceConfig}</pre>
                <Button variant="outline" size="sm" onClick={handleCopyMarketplaceConfig}>
                  <Copy className="size-4" />
                  Copy JSON
                </Button>
              </div>
            </section>
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

      {pluginToDelete && (
        <AlertDialog
          open
          onOpenChange={(open) => {
            if (!open) setPluginToDelete(null);
          }}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete Skill</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete skill: <strong>{pluginToDelete.displayName}</strong>?
              </AlertDialogDescription>
              <p className="text-sm text-muted-foreground">This action cannot be undone.</p>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <Button variant="destructive" onClick={handleDeleteConfirm} disabled={isDeleting}>
                Delete
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};

export default ClaudeCodePluginsPanel;
