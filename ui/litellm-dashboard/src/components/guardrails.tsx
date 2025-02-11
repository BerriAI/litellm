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
        setGuardrailsList(response);
      } catch (error) {
        console.error('Error fetching guardrails:', error);
      }
    };

    fetchGuardrails();
  }, [accessToken]);

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <Card>
        <Text>Configured guardrails and their current status.</Text>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Guardrail Name</TableHeaderCell>
              <TableHeaderCell>Mode</TableHeaderCell>
              <TableHeaderCell>Status</TableHeaderCell>
            </TableRow>
          </TableHead>

          <TableBody>
            {guardrailsList.map((guardrail: GuardrailItem, index: number) => (
              <TableRow key={index}>
                <TableCell>{guardrail.name}</TableCell>
                <TableCell>{guardrail.mode}</TableCell>
                <TableCell>
                  {guardrail.status === 'always_on' ? 'Always On' : 'Per Request'}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
};

export default GuardrailsPanel;