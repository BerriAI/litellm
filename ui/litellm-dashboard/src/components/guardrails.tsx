import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Card,
  Text,
  Button,
} from "@tremor/react";
import { getGuardrailsList } from "./networking";
import AddGuardrailForm from "./guardrails/add_guardrail_form";
import { PlusIcon } from "@heroicons/react/outline";
import { getGuardrailLogoAndName } from "./guardrails/guardrail_info_helpers";

interface GuardrailsPanelProps {
  accessToken: string | null;
}

interface GuardrailsResponse {
    guardrails: GuardrailItem[];
}

interface GuardrailItem {
    guardrail_name: string | null;
    litellm_params: {
        guardrail: string;
        mode: string;
        default_on: boolean;
    };
    guardrail_info: Record<string, any> | null;
}

const GuardrailsPanel: React.FC<GuardrailsPanelProps> = ({ accessToken }) => {
  const [guardrailsList, setGuardrailsList] = useState<GuardrailItem[]>([]);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const fetchGuardrails = async () => {
    if (!accessToken) {
      return;
    }
    
    setIsLoading(true);
    try {
      const response: GuardrailsResponse = await getGuardrailsList(accessToken);
      console.log(`guardrails: ${JSON.stringify(response)}`);
      setGuardrailsList(response.guardrails);
    } catch (error) {
      console.error('Error fetching guardrails:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchGuardrails();
  }, [accessToken]);

  const handleAddGuardrail = () => {
    setIsAddModalVisible(true);
  };

  const handleCloseModal = () => {
    setIsAddModalVisible(false);
  };

  const handleSuccess = () => {
    fetchGuardrails();
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex justify-between items-center mb-4">
        <Text className="text-lg">
          Configured guardrails and their current status. Setup guardrails in config.yaml or add them directly.{" "}
          <a 
              href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-blue-500 hover:text-blue-700 underline"
          >
              Docs
          </a>
        </Text>
        <Button 
          icon={PlusIcon} 
          onClick={handleAddGuardrail}
          disabled={!accessToken}
        >
          Add Guardrail
        </Button>
      </div>
      
      <Card>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Guardrail Name</TableHeaderCell>
              <TableHeaderCell>Provider</TableHeaderCell>
              <TableHeaderCell>Mode</TableHeaderCell>
              <TableHeaderCell>Status</TableHeaderCell>
            </TableRow>
          </TableHead>

          <TableBody>
            { isLoading ? (
              <TableRow>
                <TableCell colSpan={4} className="mt-4 text-gray-500 text-center py-4">Loading...</TableCell>
              </TableRow>
            ) : (!guardrailsList || guardrailsList.length === 0) ? (
              <TableRow>
                <TableCell colSpan={4} className="mt-4 text-gray-500 text-center py-4">No guardrails configured</TableCell>
              </TableRow>
            ) : (
              guardrailsList?.map((guardrail: GuardrailItem, index: number) => {
                const { logo, displayName } = getGuardrailLogoAndName(guardrail.litellm_params.guardrail);
                
                return (
                  <TableRow key={index}>
                    <TableCell>{guardrail.guardrail_name}</TableCell>
                    <TableCell>
                      <div className="flex items-center space-x-2">
                        {logo && (
                          <img 
                            src={logo} 
                            alt={`${displayName} logo`} 
                            className="w-5 h-5"
                            onError={(e) => {
                              // Hide broken image
                              (e.target as HTMLImageElement).style.display = 'none';
                            }}
                          />
                        )}
                        <span>{displayName}</span>
                      </div>
                    </TableCell>
                    <TableCell>{guardrail.litellm_params.mode}</TableCell>
                    <TableCell>
                      <div className={`inline-flex rounded-full px-2 py-1 text-xs font-medium
                          ${guardrail.litellm_params.default_on 
                          ? 'bg-green-100 text-green-800'  // Always On styling
                          : 'bg-gray-100 text-gray-800'    // Per Request styling
                          }`}>
                          {guardrail.litellm_params.default_on ? 'Always On' : 'Per Request'}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </Card>

      <AddGuardrailForm 
        visible={isAddModalVisible}
        onClose={handleCloseModal}
        accessToken={accessToken}
        onSuccess={handleSuccess}
      />
    </div>
  );
};

export default GuardrailsPanel;