import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { Pencil } from "lucide-react";
import { cn } from "@/lib/utils";

interface VariableTextAreaProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
  className?: string;
}

const VariableTextArea: React.FC<VariableTextAreaProps> = ({
  value,
  onChange,
  placeholder,
  rows = 4,
  className,
}) => {
  const [editingVariable, setEditingVariable] = useState<{
    oldName: string;
    start: number;
    end: number;
  } | null>(null);
  const [newVariableName, setNewVariableName] = useState("");

  const extractVariables = (): Array<{
    name: string;
    start: number;
    end: number;
  }> => {
    const variableRegex = /\{\{(\w+)\}\}/g;
    const variables: Array<{ name: string; start: number; end: number }> = [];
    let match;

    while ((match = variableRegex.exec(value)) !== null) {
      variables.push({
        name: match[1],
        start: match.index,
        end: match.index + match[0].length,
      });
    }

    return variables;
  };

  const handleVariableEdit = () => {
    if (!newVariableName.trim() || !editingVariable) return;

    const newValue =
      value.substring(0, editingVariable.start) +
      `{{${newVariableName}}}` +
      value.substring(editingVariable.end);

    onChange(newValue);
    setEditingVariable(null);
    setNewVariableName("");
  };

  const variables = extractVariables();

  return (
    <div className={cn("variable-textarea-container", className)}>
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        className="font-sans"
      />

      {variables.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 items-center">
          <span className="text-xs text-muted-foreground mr-1">
            Detected variables:
          </span>
          {variables.map((variable, index) => (
            <Popover
              key={`${variable.start}-${index}`}
              open={editingVariable?.start === variable.start}
              onOpenChange={(open) => {
                if (!open) {
                  setEditingVariable(null);
                  setNewVariableName("");
                }
              }}
            >
              <PopoverTrigger asChild>
                <Badge
                  className="bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300 cursor-pointer hover:opacity-80 transition-all gap-1"
                  onClick={() => {
                    setEditingVariable({
                      oldName: variable.name,
                      start: variable.start,
                      end: variable.end,
                    });
                    setNewVariableName(variable.name);
                  }}
                >
                  <Pencil className="h-3 w-3" />
                  {variable.name}
                </Badge>
              </PopoverTrigger>
              <PopoverContent className="min-w-[200px] p-2">
                <div className="text-xs text-muted-foreground mb-2">
                  Edit variable name
                </div>
                <Input
                  value={newVariableName}
                  onChange={(e) => setNewVariableName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleVariableEdit();
                    }
                  }}
                  placeholder="Variable name"
                  autoFocus
                  className="h-8"
                />
                <div className="flex gap-2 mt-2">
                  <button
                    type="button"
                    onClick={handleVariableEdit}
                    className="text-xs px-2 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90"
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setEditingVariable(null);
                      setNewVariableName("");
                    }}
                    className="text-xs px-2 py-1 bg-muted text-foreground rounded hover:bg-accent"
                  >
                    Cancel
                  </button>
                </div>
              </PopoverContent>
            </Popover>
          ))}
        </div>
      )}
    </div>
  );
};

export default VariableTextArea;
