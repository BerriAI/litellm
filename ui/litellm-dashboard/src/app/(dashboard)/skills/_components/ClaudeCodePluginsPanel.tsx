import React, { useState, useEffect } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { ArrowLeft } from "lucide-react";
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
import { getClaudeCodePluginsList, deleteClaudeCodePlugin } from "@/components/networking";
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

interface SelectedSkillProps {
  skill: Plugin | undefined;
  isLoading: boolean;
  onBack: () => void;
  isAdmin: boolean;
  accessToken: string | null;
  onPublishClick: () => void;
}

const SelectedSkill: React.FC<SelectedSkillProps> = ({ skill, isLoading, onBack, ...detailProps }) => {
  if (skill) {
    return <SkillDetail skill={skill} onBack={onBack} {...detailProps} />;
  }
  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading skill…</p>;
  }
  return (
    <div>
      <Button variant="ghost" className="mb-4" onClick={onBack}>
        <ArrowLeft />
        Back to Skills
      </Button>
      <h1 className="text-xl font-semibold">Skill not found</h1>
      <p className="text-sm text-muted-foreground">It may have been deleted.</p>
    </div>
  );
};

const ClaudeCodePluginsPanel: React.FC<ClaudeCodePluginsPanelProps> = ({ accessToken, userRole }) => {
  const [pluginsList, setPluginsList] = useState<Plugin[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);
  const [pluginToDelete, setPluginToDelete] = useState<{
    name: string;
    displayName: string;
  } | null>(null);
  const [selectedSkillId, setSelectedSkillId] = useQueryState("skill", parseAsString.withOptions({ history: "push" }));
  const selectedSkill = pluginsList.find((plugin) => plugin.id === selectedSkillId);

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
      {selectedSkillId ? (
        <SelectedSkill
          skill={selectedSkill}
          isLoading={isLoading}
          onBack={() => void setSelectedSkillId(null)}
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
            onPluginClick={(id) => void setSelectedSkillId(id)}
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
