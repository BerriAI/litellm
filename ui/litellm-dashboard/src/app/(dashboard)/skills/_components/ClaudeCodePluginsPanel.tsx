import React, { useState, useEffect } from "react";
import { Trans, useTranslation } from "react-i18next";
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

const ClaudeCodePluginsPanel: React.FC<ClaudeCodePluginsPanelProps> = ({ accessToken, userRole }) => {
  const { t } = useTranslation();
  const [pluginsList, setPluginsList] = useState<Plugin[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);
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

  const handleDeleteClick = (pluginName: string, displayName: string) => {
    setPluginToDelete({ name: pluginName, displayName });
  };

  const handleDeleteConfirm = async () => {
    if (!pluginToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deleteClaudeCodePlugin(accessToken, pluginToDelete.name);
      toast.success(t("skills.panel.toasts.deleted", { name: pluginToDelete.displayName }));
      fetchPlugins();
    } catch (error) {
      console.error("Error deleting skill:", error);
      toast.error(t("skills.panel.toasts.deleteFailed"));
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
            <h1 className="text-2xl font-bold">{t("skills.panel.title")}</h1>
            <p className="text-sm text-muted-foreground">
              <Trans
                i18nKey="skills.panel.intro"
                components={{
                  endpoint: <code className="bg-muted px-1 rounded-sm">/claude-code/marketplace.json</code>,
                }}
              />
            </p>
            <div className="mt-2 flex gap-2">
              <Button onClick={() => setIsAddModalVisible(true)} disabled={!accessToken || !isAdmin}>
                {t("skills.panel.addButton")}
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
              <AlertDialogTitle>{t("skills.panel.deleteDialog.title")}</AlertDialogTitle>
              <AlertDialogDescription>
                <Trans
                  i18nKey="skills.panel.deleteDialog.confirm"
                  components={{ name: <strong>{pluginToDelete.displayName}</strong> }}
                />
              </AlertDialogDescription>
              <p className="text-sm text-muted-foreground">{t("skills.panel.deleteDialog.irreversible")}</p>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("skills.panel.deleteDialog.cancel")}</AlertDialogCancel>
              <Button variant="destructive" onClick={handleDeleteConfirm} disabled={isDeleting}>
                {t("skills.panel.deleteDialog.delete")}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};

export default ClaudeCodePluginsPanel;
