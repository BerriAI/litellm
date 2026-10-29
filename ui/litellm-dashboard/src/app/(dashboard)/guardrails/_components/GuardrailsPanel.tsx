import { parseAsString, useQueryState } from "nuqs";
import React, { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChevronDown, Code, Plus } from "lucide-react";
import { getGuardrailsList } from "@/components/networking";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import AddGuardrailForm from "./add_guardrail_form";
import GuardrailTable from "./guardrail_table";
import { isAdminRole } from "@/utils/roles";
import GuardrailInfoView from "./guardrail_info";
import GuardrailTestPlayground from "./GuardrailTestPlayground";
import { Guardrail } from "@/components/guardrails/types";
import { GuardrailDeleteModal } from "./GuardrailDeleteModal";
import { CustomCodeModal } from "./custom_code";
import GuardrailGarden from "./guardrail_garden";
import { TeamGuardrailsTab } from "./TeamGuardrailsTab";

interface GuardrailsPanelProps {
  accessToken: string | null;
  userRole?: string;
}

interface GuardrailsResponse {
  guardrails: Guardrail[];
}

const GuardrailsPanel: React.FC<GuardrailsPanelProps> = ({ accessToken, userRole }) => {
  const [guardrailsList, setGuardrailsList] = useState<Guardrail[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isCustomCodeModalVisible, setIsCustomCodeModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [guardrailToDelete, setGuardrailToDelete] = useState<Guardrail | null>(null);
  const [selectedGuardrailId, setSelectedGuardrailId] = useQueryState(
    "guardrail",
    parseAsString.withOptions({ history: "push" }),
  );
  const isAdmin = userRole ? isAdminRole(userRole) : false;

  const fetchGuardrails = async () => {
    if (!accessToken) {
      return;
    }

    setIsLoading(true);
    try {
      const response: GuardrailsResponse = await getGuardrailsList(accessToken);
      setGuardrailsList(response.guardrails);
    } catch (error) {
      console.error("Error fetching guardrails:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchGuardrails();
  }, [accessToken]);

  const closeGuardrailDetail = () => {
    void setSelectedGuardrailId(null, { history: "replace" });
  };

  const handleAddGuardrail = () => {
    if (selectedGuardrailId) {
      closeGuardrailDetail();
    }
    setIsAddModalVisible(true);
  };

  const handleAddCustomCodeGuardrail = () => {
    if (selectedGuardrailId) {
      closeGuardrailDetail();
    }
    setIsCustomCodeModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleCloseCustomCodeModal = () => {
    setIsCustomCodeModalVisible(false);
  };

  const handleSuccess = () => {
    fetchGuardrails();
  };

  const handleDeleteClick = (guardrailId: string) => {
    setGuardrailToDelete(guardrailsList.find((g) => g.guardrail_id === guardrailId) || null);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <Tabs defaultValue="guardrails">
        <TabsList variant="line">
          {isAdmin && (
            <>
              <TabsTrigger value="garden" className="flex-none">
                Guardrail Garden
              </TabsTrigger>
              <TabsTrigger value="guardrails" className="flex-none">
                Guardrails
              </TabsTrigger>
              <TabsTrigger value="playground" className="flex-none" disabled={!accessToken}>
                Test Playground
              </TabsTrigger>
            </>
          )}
          <TabsTrigger value="submitted" className="flex-none">
            Submitted Guardrails
          </TabsTrigger>
        </TabsList>

        {isAdmin && (
          <>
            <TabsContent value="garden" keepMounted>
              <GuardrailGarden accessToken={accessToken} onGuardrailCreated={handleSuccess} />
            </TabsContent>

            <TabsContent value="guardrails" keepMounted>
              <div className="flex justify-between items-center mb-4">
                <DropdownMenu>
                  <DropdownMenuTrigger disabled={!accessToken} className={cn(buttonVariants({ variant: "default" }))}>
                    <Plus />
                    Add New Guardrail
                    <ChevronDown />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="w-56">
                    <DropdownMenuItem onClick={handleAddGuardrail}>
                      <Plus />
                      Add Provider Guardrail
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={handleAddCustomCodeGuardrail}>
                      <Code />
                      Create Custom Code Guardrail
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {selectedGuardrailId ? (
                <GuardrailInfoView
                  guardrailId={selectedGuardrailId}
                  onClose={closeGuardrailDetail}
                  onDeleted={() => {
                    closeGuardrailDetail();
                    fetchGuardrails();
                  }}
                  accessToken={accessToken}
                  isAdmin={isAdmin}
                />
              ) : (
                <GuardrailTable
                  guardrailsList={guardrailsList}
                  isLoading={isLoading}
                  onDeleteClick={handleDeleteClick}
                  onGuardrailClick={(id) => void setSelectedGuardrailId(id)}
                />
              )}

              <AddGuardrailForm
                visible={isAddModalVisible}
                onClose={handleCloseModal}
                accessToken={accessToken}
                onSuccess={handleSuccess}
              />

              <CustomCodeModal
                visible={isCustomCodeModalVisible}
                onClose={handleCloseCustomCodeModal}
                accessToken={accessToken}
                onSuccess={handleSuccess}
              />

              <GuardrailDeleteModal
                guardrail={guardrailToDelete}
                accessToken={accessToken}
                onClose={() => setGuardrailToDelete(null)}
                onDeleted={fetchGuardrails}
              />
            </TabsContent>

            <TabsContent value="playground" keepMounted>
              <GuardrailTestPlayground
                guardrailsList={guardrailsList}
                isLoading={isLoading}
                accessToken={accessToken}
                onClose={() => {}}
              />
            </TabsContent>
          </>
        )}

        <TabsContent value="submitted" keepMounted>
          <TeamGuardrailsTab accessToken={accessToken} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default GuardrailsPanel;
