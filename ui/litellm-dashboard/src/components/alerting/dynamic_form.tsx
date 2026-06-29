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

  const onFinish = () => {
    const formData = form.getFieldsValue();
    const isEmpty = Object.entries(formData).every(([key, value]) => {
      if (typeof value === "boolean") {
        return false;
      }
      return value === "" || value === null || value === undefined;
    });
    if (!isEmpty) {
      const converted = Object.fromEntries(
        Object.entries(formData).map(([key, val]) => {
          const setting = alertingSettings.find((s) => s.field_name === key);
          if (setting?.field_type === "List" && typeof val === "string" && val.trim() !== "") {
            const parsed = val
              .split(",")
              .map((s: string) => parseFloat(s.trim()))
              .filter((n: number) => !isNaN(n));
            return [key, parsed];
          }
          return [key, val];
        }),
      );
      handleSubmit(converted);
    }
  };

  const listDisplayValue = (fieldValue: unknown): string => {
    if (Array.isArray(fieldValue)) return fieldValue.join(", ");
    if (typeof fieldValue === "string") return fieldValue;
    return "";
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
          {value.premium_field ? (
            premiumUser ? (
              <Form.Item name={value.field_name}>
                <TableCell>
                  {value.field_type === "Integer" ? (
                    <InputNumber
                      step={1}
                      value={value.field_value}
                      onChange={(e) => handleInputChange(value.field_name, e)}
                    />
                  ) : value.field_type === "Boolean" ? (
                    <Switch
                      checked={value.field_value}
                      onChange={(checked) => handleInputChange(value.field_name, checked)}
                    />
                  ) : value.field_type === "List" ? (
                    <Input
                      value={listDisplayValue(value.field_value)}
                      onChange={(e) => handleInputChange(value.field_name, e.target.value)}
                      placeholder="e.g. 0.8, 0.85, 0.95"
                    />
                  ) : (
                    <Input value={value.field_value} onChange={(e) => handleInputChange(value.field_name, e)} />
                  )}
                </TableCell>
              </Form.Item>
            ) : (
              <TableCell>
                <Button className="flex items-center justify-center">
                  <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                    ✨ Enterprise Feature
                  </a>
                </Button>
              </TableCell>
            )
          ) : (
            <Form.Item
              name={value.field_name}
              className="mb-0"
              valuePropName={value.field_type === "Boolean" ? "checked" : "value"}
            >
              <TableCell>
                {value.field_type === "Integer" ? (
                  <InputNumber
                    step={1}
                    value={value.field_value}
                    onChange={(e) => handleInputChange(value.field_name, e)}
                    className="p-0"
                  />
                ) : value.field_type === "Boolean" ? (
                  <Switch
                    checked={value.field_value}
                    onChange={(checked) => {
                      handleInputChange(value.field_name, checked);
                      form.setFieldsValue({ [value.field_name]: checked });
                    }}
                  />
                ) : value.field_type === "List" ? (
                  <Input
                    value={listDisplayValue(value.field_value)}
                    onChange={(e) => handleInputChange(value.field_name, e.target.value)}
                    placeholder="e.g. 0.8, 0.85, 0.95"
                  />
                ) : (
                  <Input value={value.field_value} onChange={(e) => handleInputChange(value.field_name, e)} />
                )}
              </TableCell>
            </Form.Item>
          )}
          <TableCell>
            {value.stored_in_db == true ? (
              <Badge icon={CheckCircleIcon} className="text-white">
                In DB
              </Badge>
            ) : value.stored_in_db == false ? (
              <Badge className="text-gray bg-white outline">In Config</Badge>
            ) : (
              <Badge className="text-gray bg-white outline">Not Set</Badge>
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
        <Button2 htmlType="submit">Update Settings</Button2>
      </div>
    </Form>
  );
};

export default DynamicForm;
