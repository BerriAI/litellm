import React from "react";
import { Form, Input, InputNumber, Button as Button2 } from "antd";
import { CircleCheck, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { TableCell, TableRow } from "@/components/ui/table";
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
        return false; // Boolean values are never considered empty
      }
      return value === "" || value === null || value === undefined;
    });
    if (!isEmpty) {
      handleSubmit(formData);
    } else {
    }
  };

  return (
    <Form form={form} onFinish={onFinish} labelAlign="left">
      {alertingSettings.map((value, index) => (
        <TableRow key={index}>
          <TableCell>
            <p className="text-sm">{value.field_name}</p>
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
                      aria-label={value.field_name}
                      checked={value.field_value}
                      onCheckedChange={(checked) => handleInputChange(value.field_name, checked)}
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
                    aria-label={value.field_name}
                    checked={value.field_value}
                    onCheckedChange={(checked) => {
                      handleInputChange(value.field_name, checked);
                      form.setFieldsValue({ [value.field_name]: checked });
                    }}
                  />
                ) : (
                  <Input value={value.field_value} onChange={(e) => handleInputChange(value.field_name, e)} />
                )}
              </TableCell>
            </Form.Item>
          )}
          <TableCell>
            {value.stored_in_db == true ? (
              <Badge variant="secondary">
                <CircleCheck />
                In DB
              </Badge>
            ) : value.stored_in_db == false ? (
              <Badge variant="outline">In Config</Badge>
            ) : (
              <Badge variant="outline">Not Set</Badge>
            )}
          </TableCell>
          <TableCell>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Reset ${value.field_name}`}
              onClick={() => handleResetField(value.field_name, index)}
              className="text-red-500"
            >
              <Trash2 className="size-5" />
            </Button>
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
