import React from "react";
import { Form, Input, InputNumber, Button as Button2 } from "antd";
import { TrashIcon, CheckCircleIcon } from "@heroicons/react/outline";
import { Button, Badge, Icon, Text, TableRow, TableCell, Switch } from "@tremor/react";
interface AlertingSetting {
  field_name: string;
  field_description: string;
  field_type: string;
  field_value: any;
  stored_in_db: boolean | null;
  premium_field: boolean;
}

interface DynamicFormProps {
  alertingSettings: AlertingSetting[];
  handleInputChange: (fieldName: string, newValue: any) => void;
  handleResetField: (fieldName: string, index: number) => void;
  handleSubmit: (formValues: Record<string, any>) => void;
  premiumUser: boolean;
}

const DynamicForm: React.FC<DynamicFormProps> = ({
  alertingSettings,
  handleInputChange,
  handleResetField,
  handleSubmit,
  premiumUser,
}) => {
  const [form] = Form.useForm();

  // Inputs are controlled by `alertingSettings`, so that state - not the antd form
  // store - is what gets submitted. A "List" field is edited as a comma-separated
  // string, so parse it back to numbers here.
  const submitValue = (setting: AlertingSetting): any => {
    if (setting.field_type === "List") {
      if (Array.isArray(setting.field_value)) return setting.field_value;
      if (typeof setting.field_value !== "string") return [];
      return setting.field_value
        .split(",")
        .map((s: string) => parseFloat(s.trim()))
        .filter((n: number) => !isNaN(n));
    }
    if (setting.field_type === "String") {
      const trimmed = typeof setting.field_value === "string" ? setting.field_value.trim() : setting.field_value;
      return trimmed === "" || trimmed === undefined ? null : trimmed;
    }
    return setting.field_value;
  };

  const onFinish = () => {
    if (alertingSettings.length === 0) {
      return;
    }
    handleSubmit(Object.fromEntries(alertingSettings.map((setting) => [setting.field_name, submitValue(setting)])));
  };

  const listDisplayValue = (fieldValue: unknown): string => {
    if (Array.isArray(fieldValue)) return fieldValue.join(", ");
    if (typeof fieldValue === "string") return fieldValue;
    return "";
  };

  const renderFieldInput = (value: AlertingSetting): React.ReactNode => {
    if (value.field_type === "Integer") {
      return (
        <InputNumber
          step={1}
          value={value.field_value}
          onChange={(e) => handleInputChange(value.field_name, e)}
          className="p-0"
        />
      );
    }
    if (value.field_type === "Boolean") {
      return (
        <Switch checked={value.field_value} onChange={(checked) => handleInputChange(value.field_name, checked)} />
      );
    }
    if (value.field_type === "List") {
      return (
        <Input
          value={listDisplayValue(value.field_value)}
          onChange={(e) => handleInputChange(value.field_name, e.target.value)}
          placeholder="e.g. 0.8, 0.85, 0.95"
        />
      );
    }
    return (
      <Input
        value={value.field_value ?? ""}
        onChange={(e) => handleInputChange(value.field_name, e.target.value)}
        placeholder={value.field_name.endsWith("_url") ? "https://example.com/webhook" : ""}
      />
    );
  };

  const renderSettingCell = (value: AlertingSetting): React.ReactNode => {
    if (value.premium_field && !premiumUser) {
      return (
        <TableCell>
          <Button className="flex items-center justify-center">
            <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
              ✨ Enterprise Feature
            </a>
          </Button>
        </TableCell>
      );
    }
    return <TableCell>{renderFieldInput(value)}</TableCell>;
  };

  const renderStoredBadge = (storedInDb: boolean | null): React.ReactNode => {
    if (storedInDb === true) {
      return (
        <Badge icon={CheckCircleIcon} className="text-white">
          In DB
        </Badge>
      );
    }
    if (storedInDb === false) {
      return <Badge className="text-gray bg-white outline-solid">In Config</Badge>;
    }
    return <Badge className="text-gray bg-white outline-solid">Not Set</Badge>;
  };

  return (
    <Form form={form} onFinish={onFinish} labelAlign="left">
      {alertingSettings.map((value, index) => (
        <TableRow key={index}>
          <TableCell align="center">
            <Text>{value.field_name}</Text>
            <p
              style={{
                fontSize: "0.65rem",
                color: "#808080",
                fontStyle: "italic",
              }}
              className="mt-1"
            >
              {value.field_description}
            </p>
          </TableCell>
          {renderSettingCell(value)}
          <TableCell>{renderStoredBadge(value.stored_in_db)}</TableCell>
          <TableCell>
            <Icon icon={TrashIcon} color="red" onClick={() => handleResetField(value.field_name, index)}>
              Reset
            </Icon>
          </TableCell>
        </TableRow>
      ))}
      <div>
        <Button2 htmlType="submit">Update Settings</Button2>
      </div>
    </Form>
  );
};

export default DynamicForm;
