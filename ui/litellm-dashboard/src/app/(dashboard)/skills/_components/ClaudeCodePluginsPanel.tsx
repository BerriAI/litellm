import React, { useState, useEffect, useMemo } from "react";
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
import { getClaudeCodePluginsList, deleteClaudeCodePlugin, reviewClaudeCodePlugin } from "@/components/networking";
import AddPluginForm from "./add_plugin_form";
import PluginTable from "./PluginTable";
import ReviewSkillDialog, { ReviewDecision } from "./ReviewSkillDialog";
import { countAwaitingReview, isAwaitingReview, reviewFailureMessage } from "@/components/claude_code_plugins/helpers";
import SkillDetail from "@/components/claude_code_plugins/skill_detail";
import { isAdminRole } from "@/utils/roles";
import NotificationsManager from "@/components/molecules/notifications_manager";
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
  const [pluginToDelete, setPluginToDelete] = useState<{
    name: string;
    displayName: string;
  } | null>(null);
  const [selectedSkill, setSelectedSkill] = useState<Plugin | null>(null);
  const [pluginToReview, setPluginToReview] = useState<{ plugin: Plugin; decision: ReviewDecision } | null>(null);
  const [isReviewing, setIsReviewing] = useState(false);
  const [showPendingOnly, setShowPendingOnly] = useState(false);

  const isAdmin = userRole ? isAdminRole(userRole) : false;
  const pendingCount = useMemo(() => countAwaitingReview(pluginsList), [pluginsList]);
  const visiblePlugins = useMemo(
    () => (showPendingOnly ? pluginsList.filter(isAwaitingReview) : pluginsList),
    [pluginsList, showPendingOnly],
  );

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

  const handleReviewConfirm = async (notes: string) => {
    if (!pluginToReview || !accessToken) return;

    setIsReviewing(true);
    try {
      await reviewClaudeCodePlugin(accessToken, pluginToReview.plugin.name, {
        decision: pluginToReview.decision,
        reviewNotes: notes,
        reviewedFingerprint: pluginToReview.plugin.manifest_fingerprint,
      });
      NotificationsManager.success(
        pluginToReview.decision === "approve"
          ? `Skill "${pluginToReview.plugin.name}" approved and published`
          : `Skill "${pluginToReview.plugin.name}" rejected`,
      );
      fetchPlugins();
    } catch (error) {
      console.error("Error reviewing skill:", error);
      NotificationsManager.error(reviewFailureMessage(error));
      fetchPlugins();
    } finally {
      setIsReviewing(false);
      setPluginToReview(null);
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
            <p className="text-sm text-gray-600">
              {isAdmin
                ? "Register Claude Code skills and review skills submitted by your users. Approved skills appear in the Skill Hub for all users and are served via "
                : "Submit Claude Code skills for administrator review. Once approved, a skill appears in the Skill Hub for all users and is served via "}
              <code className="bg-gray-100 px-1 rounded-sm">/claude-code/marketplace.json</code>.
            </p>
            <div className="mt-2 flex gap-2">
              <Button onClick={() => setIsAddModalVisible(true)} disabled={!accessToken}>
                {isAdmin ? "+ Add Skill" : "+ Submit Skill"}
              </Button>
              {isAdmin && pendingCount > 0 && (
                <Button
                  variant={showPendingOnly ? "default" : "secondary"}
                  data-testid="toggle-pending-review"
                  onClick={() => setShowPendingOnly(!showPendingOnly)}
                >
                  {`Awaiting review (${pendingCount})`}
                </Button>
              )}
            </div>
          </div>

          <PluginTable
            pluginsList={visiblePlugins}
            isLoading={isLoading}
            onDeleteClick={handleDeleteClick}
            onReviewClick={(plugin, decision) => setPluginToReview({ plugin, decision })}
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

      {pluginToReview && (
        <ReviewSkillDialog
          skillName={pluginToReview.plugin.name}
          decision={pluginToReview.decision}
          isSubmitting={isReviewing}
          onCancel={() => setPluginToReview(null)}
          onConfirm={handleReviewConfirm}
        />
      )}

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
