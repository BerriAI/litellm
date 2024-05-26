import React from "react";
import { Form, Input, InputNumber, Row, Col, Button as Button2 } from "antd";
import { TrashIcon, CheckCircleIcon } from "@heroicons/react/outline";
import { Button, Badge, Icon, Text, TableRow, TableCell } from "@tremor/react";
import Paragraph from "antd/es/typography/Paragraph";
interface AlertingSetting {
  field_name: string;
  field_description: string;
  field_type: string;
  field_value: any;
  stored_in_db: boolean | null;
}

interface DynamicFormProps {
  alertingSettings: AlertingSetting[];
  handleInputChange: (fieldName: string, newValue: any) => void;
  handleResetField: (fieldName: string, index: number) => void;
  handleSubmit: (formValues: Record<string, any>) => void;
}

const DynamicForm: React.FC<DynamicFormProps> = ({
  alertingSettings,
  handleInputChange,
  handleResetField,
  handleSubmit,
}) => {
  const [form] = Form.useForm();

  const onFinish = () => {
    const formData = form.getFieldsValue();
    handleSubmit(formData);
  };

  return (
    <Form form={form} onFinish={onFinish} labelAlign="left">
      {alertingSettings.map((value, index) => (
        <TableRow key={index}>
          <TableCell>
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
          <Form.Item name={value.field_name}>
            <TableCell>
              {value.field_type === "Integer" ? (
                <InputNumber
                  step={1}
                  value={value.field_value}
                  onChange={(e) => handleInputChange(value.field_name, e)}
                />
              ) : (
                <Input
                  value={value.field_value}
                  onChange={(e) => handleInputChange(value.field_name, e)}
                />
              )}
            </TableCell>
          </Form.Item>
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
            <Icon
              icon={TrashIcon}
              color="red"
              onClick={() => handleResetField(value.field_name, index)}
            >
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
