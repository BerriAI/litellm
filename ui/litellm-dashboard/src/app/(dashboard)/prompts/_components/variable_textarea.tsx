import React, { useState } from "react";
import { PencilIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";

interface VariableTextAreaProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
  className?: string;
}

const VariableTextArea: React.FC<VariableTextAreaProps> = ({ value, onChange, placeholder, rows = 4, className }) => {
  const [editingVariable, setEditingVariable] = useState<{
    oldName: string;
    start: number;
    end: number;
  } | null>(null);
  const [newVariableName, setNewVariableName] = useState("");

  // Extract all variables from the text
  const extractVariables = (): Array<{ name: string; start: number; end: number }> => {
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
      value.substring(0, editingVariable.start) + `{{${newVariableName}}}` + value.substring(editingVariable.end);

    onChange(newValue);
    setEditingVariable(null);
    setNewVariableName("");
  };

  const variables = extractVariables();

  // New approach: Use ContentEditable div for true inline styling
  // This is much harder to get right with React, so for now, let's stick to the reliable
  // "Tags Below" approach which is robust and functional.
  // If user insists on inline coloring, we can revisit the overlay approach but it's very fragile.

  // BUT, to satisfy "variables in text box", we can try a simple trick:
  // Render the text as HTML with colored spans inside a contentEditable div
  // and sync it back. This is the "wysiwyg" approach.

  return (
    <div className={`variable-textarea-container ${className}`}>
      {/* Using standard TextArea for reliability */}
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        className="field-sizing-fixed font-sans"
      />

      {/* Variable Management - Clear and Functional */}
      {variables.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 items-center">
          <span className="text-xs text-muted-foreground mr-1">Detected variables:</span>
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
              <PopoverTrigger
                render={
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-auto p-0"
                    onClick={() => {
                      setEditingVariable({
                        oldName: variable.name,
                        start: variable.start,
                        end: variable.end,
                      });
                      setNewVariableName(variable.name);
                    }}
                  />
                }
              >
                <Badge variant="outline" className="cursor-pointer">
                  <PencilIcon className="size-3" />
                  {variable.name}
                </Badge>
              </PopoverTrigger>
              <PopoverContent className="w-[216px]">
                <div className="p-2">
                  <div className="text-xs text-muted-foreground mb-2">Edit variable name</div>
                  <Input
                    value={newVariableName}
                    onChange={(e) => setNewVariableName(e.target.value)}
                    onKeyDown={(event) => event.key === "Enter" && handleVariableEdit()}
                    placeholder="Variable name"
                    autoFocus
                  />
                  <div className="flex gap-2 mt-2">
                    <Button size="sm" onClick={handleVariableEdit}>
                      Save
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setEditingVariable(null);
                        setNewVariableName("");
                      }}
                    >
                      Cancel
                    </Button>
                  </div>
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
