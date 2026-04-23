import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { TableCell, TableRow } from "@/components/ui/table";
import { CheckCircle, Trash2 } from "lucide-react";

interface AlertingSetting {
  field_name: string;
  field_description: string;
  field_type: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  field_value: any;
  stored_in_db: boolean | null;
  premium_field: boolean;
}

interface DynamicFormProps {
  alertingSettings: AlertingSetting[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handleInputChange: (fieldName: string, newValue: any) => void;
  handleResetField: (fieldName: string, index: number) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handleSubmit: (formValues: Record<string, any>) => void;
  premiumUser: boolean;
}

/**
 * Alerting-settings dynamic form. Settings are rendered row-by-row into an
 * existing table context (this component emits `<TableRow>` children that
 * must be placed inside a `<Table>`/`<TableBody>` parent).
 *
 * Post phase-1 migration: antd Form replaced with a native form element
 * driving a local state bag; no rhf/zod here because the field set is
 * dynamic (driven by `alertingSettings`).
 */
const DynamicForm: React.FC<DynamicFormProps> = ({
  alertingSettings,
  handleInputChange,
  handleResetField,
  handleSubmit,
  premiumUser,
}) => {
  // Mirror the form state locally so we can submit a consolidated snapshot
  // without depending on antd's Form.getFieldsValue().
  const [formValues, setFormValues] = useState<Record<string, unknown>>({});

  const updateLocal = (name: string, value: unknown) => {
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const isEmpty = Object.entries(formValues).every(([, value]) => {
      if (typeof value === "boolean") return false;
      return value === "" || value === null || value === undefined;
    });
    if (!isEmpty) {
      handleSubmit(formValues);
    } else {
      console.log("Some form fields are empty.");
    }
  };

  return (
    <form onSubmit={onSubmit}>
      {alertingSettings.map((value, index) => {
        const renderInput = () => {
          if (value.field_type === "Integer") {
            return (
              <Input
                type="number"
                step={1}
                defaultValue={value.field_value ?? ""}
                onChange={(e) => {
                  const v = e.target.value === "" ? null : Number(e.target.value);
                  handleInputChange(value.field_name, v);
                  updateLocal(value.field_name, v);
                }}
              />
            );
          }
          if (value.field_type === "Boolean") {
            return (
              <Switch
                checked={!!value.field_value}
                onCheckedChange={(checked) => {
                  handleInputChange(value.field_name, checked);
                  updateLocal(value.field_name, checked);
                }}
              />
            );
          }
          return (
            <Input
              defaultValue={value.field_value ?? ""}
              onChange={(e) => {
                handleInputChange(value.field_name, e.target.value);
                updateLocal(value.field_name, e.target.value);
              }}
            />
          );
        };

        return (
          <TableRow key={index}>
            <TableCell>
              <p>{value.field_name}</p>
              <p className="text-[0.65rem] text-muted-foreground italic mt-1">
                {value.field_description}
              </p>
            </TableCell>
            <TableCell>
              {value.premium_field && !premiumUser ? (
                <Button asChild>
                  <a
                    href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    ✨ Enterprise Feature
                  </a>
                </Button>
              ) : (
                renderInput()
              )}
            </TableCell>
            <TableCell>
              {value.stored_in_db === true ? (
                <Badge className="gap-1">
                  <CheckCircle size={12} />
                  In DB
                </Badge>
              ) : value.stored_in_db === false ? (
                <Badge variant="outline">In Config</Badge>
              ) : (
                <Badge variant="outline">Not Set</Badge>
              )}
            </TableCell>
            <TableCell>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="text-destructive hover:text-destructive"
                onClick={() => handleResetField(value.field_name, index)}
              >
                <Trash2 size={14} />
                Reset
              </Button>
            </TableCell>
          </TableRow>
        );
      })}
      <div>
        <Button type="submit">Update Settings</Button>
      </div>
    </form>
  );
};

export default DynamicForm;
