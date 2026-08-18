import React from "react";
import { useForm } from "react-hook-form";
import { TrashIcon, CheckCircleIcon } from "@heroicons/react/outline";
import { Button, Badge, Icon, Text, TableRow, TableCell, Switch } from "@tremor/react";
import { Input } from "@/components/ui/input";

interface AlertingSetting {
  field_name: string;
  field_description: string;
  field_type: string;
  field_value: any;
  stored_in_db: boolean | null;
  premium_field: boolean;
}

export type AlertingFieldValue = string | number | boolean | null;

interface DynamicFormProps {
  alertingSettings: AlertingSetting[];
  handleInputChange: (fieldName: string, newValue: AlertingFieldValue) => void;
  handleResetField: (fieldName: string, index: number) => void;
  handleSubmit: (formValues: Record<string, any>) => void;
  premiumUser: boolean;
}

type AlertingFormValues = Record<string, string | boolean>;

const DynamicForm: React.FC<DynamicFormProps> = ({
  alertingSettings,
  handleInputChange,
  handleResetField,
  handleSubmit,
  premiumUser,
}) => {
  const form = useForm<AlertingFormValues>({ defaultValues: {} });

  const onFinish = (formData: AlertingFormValues) => {
    const isEmpty = Object.entries(formData).every(([, value]) => {
      if (typeof value === "boolean") {
        return false;
      }
      return value === "" || value === null || value === undefined;
    });
    if (!isEmpty) {
      handleSubmit(formData);
    }
  };

  const handleNumericChange = (setting: AlertingSetting, raw: string) => {
    form.setValue(setting.field_name, raw);
    handleInputChange(setting.field_name, raw === "" ? null : Number(raw));
  };

  const handleTextChange = (setting: AlertingSetting, event: React.ChangeEvent<HTMLInputElement>) => {
    form.setValue(setting.field_name, event.target.value);
    handleInputChange(setting.field_name, event.target.value);
  };

  const handleToggle = (setting: AlertingSetting, checked: boolean) => {
    form.setValue(setting.field_name, checked);
    handleInputChange(setting.field_name, checked);
  };

  const renderControl = (setting: AlertingSetting) => {
    if (setting.field_type === "Integer") {
      return (
        <Input
          type="number"
          step={1}
          value={setting.field_value ?? ""}
          onChange={(event) => handleNumericChange(setting, event.target.value)}
        />
      );
    }
    if (setting.field_type === "Boolean") {
      return <Switch checked={setting.field_value} onChange={(checked) => handleToggle(setting, checked)} />;
    }
    return <Input value={setting.field_value ?? ""} onChange={(event) => handleTextChange(setting, event)} />;
  };

  return (
    <form onSubmit={form.handleSubmit(onFinish)} noValidate>
      {alertingSettings.map((value, index) => (
        <TableRow key={index}>
          <TableCell align="center">
            <Text>{value.field_name}</Text>
            <p className="mt-1 text-[0.65rem] italic text-muted-foreground">{value.field_description}</p>
          </TableCell>
          {value.premium_field && !premiumUser ? (
            <TableCell>
              <Button className="flex items-center justify-center">
                <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                  ✨ Enterprise Feature
                </a>
              </Button>
            </TableCell>
          ) : (
            <TableCell>{renderControl(value)}</TableCell>
          )}
          <TableCell>
            {value.stored_in_db == true ? (
              <Badge icon={CheckCircleIcon} className="text-white">
                In DB
              </Badge>
            ) : value.stored_in_db == false ? (
              <Badge className="text-gray bg-background outline-solid">In Config</Badge>
            ) : (
              <Badge className="text-gray bg-background outline-solid">Not Set</Badge>
            )}
          </TableCell>
          <TableCell>
            <Icon icon={TrashIcon} color="red" onClick={() => handleResetField(value.field_name, index)}>
              Reset
            </Icon>
          </TableCell>
        </TableRow>
      ))}
      <div>
        <Button type="submit">Update Settings</Button>
      </div>
    </form>
  );
};

export default DynamicForm;
