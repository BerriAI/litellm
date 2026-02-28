import React from "react";
import { GuardrailsPage } from "./guardrails/GuardrailsPage";

interface GuardrailsPanelProps {
  accessToken: string | null;
  userRole?: string;
}

const GuardrailsPanel: React.FC<GuardrailsPanelProps> = ({ accessToken }) => {
  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <GuardrailsPage accessToken={accessToken} />
    </div>
  );
};

export default GuardrailsPanel;
