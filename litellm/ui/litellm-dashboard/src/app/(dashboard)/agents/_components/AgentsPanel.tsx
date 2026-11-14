import { useTerminology } from "@/contexts/TerminologyContext";
import React, { useState, useEffect } from "react";
import { Info, Plus } from "lucide-react";
import { useTranslation } from "react-i18next";
import { getAgentsList, deleteAgentCall } from "@/components/networking";
import AddAgentForm from "./add_agent_form";
import { isAdminRole } from "@/utils/roles";
import AgentInfoView from "./agent_info";
import AgentsTable from "./AgentsTable";
import { toast } from "@/lib/toast";
import { Agent } from "@/components/agents/types";
import { Team } from "@/components/key_team_helpers/key_list";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";

interface AgentsPanelProps {
  accessToken: string | null;
  userRole?: string;
  teams?: Team[] | null;
}

interface AgentsResponse {
  agents: Agent[];
}

const AgentsPanel: React.FC<AgentsPanelProps> = ({ accessToken, userRole, teams }) => {
  const term = useTerminology();
  const { t } = useTranslation();
  const [agentsList, setAgentsList] = useState<Agent[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isHealthCheckLoading, setIsHealthCheckLoading] = useState(false);
  const [agentToDelete, setAgentToDelete] = useState<{ id: string; name: string } | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [healthCheckEnabled, setHealthCheckEnabled] = useState(false);

  const isAdmin = userRole ? isAdminRole(userRole) : false;

  useEffect(() => {
    let cancelled = false;
    const loadForToken = async () => {
      if (!accessToken) {
        setAgentsList([]);
        setIsLoading(false);
        return;
      }
      setIsLoading(true);
      try {
        const response: AgentsResponse = await getAgentsList(accessToken, false);
        if (!cancelled) {
          setAgentsList(response.agents || []);
        }
      } catch (error) {
        console.error("Error fetching agents:", error);
        if (!cancelled) {
          setAgentsList([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };
    loadForToken();
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  const refetchAgents = async (healthCheck: boolean) => {
    if (!accessToken) {
      return;
    }
    try {
      const response: AgentsResponse = await getAgentsList(accessToken, healthCheck);
      setAgentsList(response.agents || []);
    } catch (error) {
      console.error("Error fetching agents:", error);
    }
  };

  const handleHealthCheckToggle = async (checked: boolean) => {
    setHealthCheckEnabled(checked);
    setIsHealthCheckLoading(true);
    try {
      await refetchAgents(checked);
    } finally {
      setIsHealthCheckLoading(false);
    }
  };

  const handleAddAgent = () => {
    if (selectedAgentId) {
      setSelectedAgentId(null);
    }
    setIsAddModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleSuccess = () => {
    refetchAgents(healthCheckEnabled);
  };

  const handleDeleteClick = (agentId: string, agentName: string) => {
    setAgentToDelete({ id: agentId, name: agentName });
  };

  const handleDeleteConfirm = async () => {
    if (!agentToDelete || !accessToken) return;

    setIsDeleting(true);
    try {
      await deleteAgentCall(accessToken, agentToDelete.id);
      toast.success(t("agents.panel.toasts.deleted", { name: agentToDelete.name }));
      await refetchAgents(healthCheckEnabled);
    } catch (error) {
      console.error("Error deleting agent:", error);
      toast.fromError(t("agents.panel.toasts.deleteFailed"));
    } finally {
      setIsDeleting(false);
      setAgentToDelete(null);
    }
  };

  const handleDeleteCancel = () => {
    setAgentToDelete(null);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex flex-col gap-2 mb-4">
        <h1 className="text-2xl font-bold">{t("agents.panel.title")}</h1>
        <p className="text-sm text-muted-foreground">{term(t("agents.panel.description"))}</p>
        <Alert className="mb-3">
          <Info />
          <AlertTitle>{t("agents.panel.authAlert.title")}</AlertTitle>
          <AlertDescription>{t("agents.panel.authAlert.description")}</AlertDescription>
        </Alert>
        {isAdmin && (
          <div className="mt-2 flex items-center gap-4">
            <Button onClick={handleAddAgent} disabled={!accessToken}>
              <Plus />
              {t("agents.panel.addAgent")}
            </Button>
          </div>
        )}
      </div>

      {selectedAgentId ? (
        <AgentInfoView
          agentId={selectedAgentId}
          onClose={() => setSelectedAgentId(null)}
          accessToken={accessToken}
          isAdmin={isAdmin}
        />
      ) : (
        <AgentsTable
          agents={agentsList}
          isLoading={isLoading}
          isAdmin={isAdmin}
          healthCheckEnabled={healthCheckEnabled}
          isHealthCheckLoading={isHealthCheckLoading}
          onHealthCheckToggle={handleHealthCheckToggle}
          onAgentClick={(id) => setSelectedAgentId(id)}
          onDeleteClick={handleDeleteClick}
        />
      )}

      <AddAgentForm
        visible={isAddModalVisible}
        onClose={handleCloseModal}
        accessToken={accessToken}
        onSuccess={handleSuccess}
        teams={teams}
      />

      {agentToDelete && (
        <AlertDialog
          open
          onOpenChange={(open) => {
            if (!open) handleDeleteCancel();
          }}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t("agents.panel.deleteDialog.title")}</AlertDialogTitle>
              <AlertDialogDescription>
                {t("agents.panel.deleteDialog.description", { name: agentToDelete.name })}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("agents.panel.deleteDialog.cancel")}</AlertDialogCancel>
              <Button variant="destructive" onClick={handleDeleteConfirm} disabled={isDeleting}>
                {t("agents.panel.deleteDialog.confirm")}
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
};

export default AgentsPanel;
