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
} from "@tremor/react";
import { getGuardrailsList } from "./networking";

interface GuardrailsPanelProps {
  accessToken: string | null;
}

interface GuardrailsResponse {
    guardrails: GuardrailItem[];
}

interface GuardrailItem {
  name: string;
  mode: string;
  status: "always_on" | "per_request";
}

const GuardrailsPanel: React.FC<GuardrailsPanelProps> = ({ accessToken }) => {
  const [guardrailsList, setGuardrailsList] = useState<GuardrailItem[]>([]);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    
    const fetchGuardrails = async () => {
      try {
        const response = await getGuardrailsList(accessToken);
        const data: GuardrailsResponse = await response.json();

        setGuardrailsList(data.guardrails);

      } catch (error) {
        console.error('Error fetching guardrails:', error);
      }
    };

    fetchGuardrails();
  }, [accessToken]);

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
        <Text className="mb-4">
        Configured guardrails and their current status. Setup guardrails in config.yaml.{" "}
        <a 
            href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-blue-500 hover:text-blue-700 underline"
        >
            Docs
        </a>
        </Text>
      <Card>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Guardrail Name</TableHeaderCell>
              <TableHeaderCell>Mode</TableHeaderCell>
              <TableHeaderCell>Status</TableHeaderCell>
            </TableRow>
          </TableHead>

          <TableBody>
            { (!guardrailsList || guardrailsList.length === 0) ? (
              <TableRow>
                <TableCell colSpan={3} className="mt-4 text-gray-500 text-center py-4">No guardrails configured</TableCell>
              </TableRow>
            ) : (
              guardrailsList?.map((guardrail: GuardrailItem, index: number) => (
                <TableRow key={index}>
                <TableCell>{guardrail.name}</TableCell>
                <TableCell>{guardrail.mode}</TableCell>
                <TableCell>
                  {guardrail.status === 'always_on' ? 'Always On' : 'Per Request'}
                </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
};

export default GuardrailsPanel;