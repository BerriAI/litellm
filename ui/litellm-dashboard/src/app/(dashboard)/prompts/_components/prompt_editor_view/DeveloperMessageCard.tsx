import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import VariableTextArea from "../variable_textarea";

interface DeveloperMessageCardProps {
  value: string;
  onChange: (value: string) => void;
}

const DeveloperMessageCard: React.FC<DeveloperMessageCardProps> = ({ value, onChange }) => {
  return (
    <Card>
      <CardContent className="p-3">
        <p className="mb-2 text-sm font-medium text-foreground">Developer message</p>
        <p className="mb-2 text-xs text-muted-foreground">Optional system instructions for the model</p>
        <VariableTextArea
          value={value}
          onChange={onChange}
          rows={3}
          placeholder="e.g., You are a helpful assistant..."
        />
      </CardContent>
    </Card>
  );
};

export default DeveloperMessageCard;
