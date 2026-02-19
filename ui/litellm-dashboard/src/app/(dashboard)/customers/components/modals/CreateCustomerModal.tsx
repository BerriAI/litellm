import React, { useState } from "react";
import { Modal, Form, Input, InputNumber, Select as AntSelect } from "antd";
import type { NewCustomerData } from "@/app/(dashboard)/customers/types";

interface CreateCustomerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (customer: NewCustomerData) => void;
}

const CreateCustomerModal: React.FC<CreateCustomerModalProps> = ({
  isOpen,
  onClose,
  onCreate,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  const handleOk = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      onCreate(values);
      form.resetFields();
    } catch (error) {
      console.error("Validation failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onClose();
  };

  return (
    <Modal
      title="Create New Customer"
      open={isOpen}
      onOk={handleOk}
      onCancel={handleCancel}
      confirmLoading={loading}
      okText="Create Customer"
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        className="mt-4"
      >
        <Form.Item
          label="User ID"
          name="user_id"
          rules={[{ required: true, message: "Please enter a user ID" }]}
        >
          <Input placeholder="e.g. customer-007" />
        </Form.Item>

        <Form.Item
          label="Alias"
          name="alias"
        >
          <Input placeholder="e.g. Acme Corp" />
        </Form.Item>

        <div className="grid grid-cols-2 gap-4">
          <Form.Item
            label="Max Budget"
            name="max_budget"
          >
            <InputNumber
              className="w-full"
              placeholder="e.g. 500"
              min={0}
            />
          </Form.Item>

          <Form.Item
            label="Budget ID"
            name="budget_id"
          >
            <Input placeholder="e.g. free_tier" />
          </Form.Item>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Form.Item
            label="Default Model"
            name="default_model"
          >
            <AntSelect placeholder="Select model...">
              <AntSelect.Option value="gpt-4o">gpt-4o</AntSelect.Option>
              <AntSelect.Option value="gpt-4o-mini">gpt-4o-mini</AntSelect.Option>
              <AntSelect.Option value="gpt-4-turbo">gpt-4-turbo</AntSelect.Option>
              <AntSelect.Option value="claude-3-sonnet">claude-3-sonnet</AntSelect.Option>
              <AntSelect.Option value="claude-3-opus">claude-3-opus</AntSelect.Option>
              <AntSelect.Option value="claude-3-haiku">claude-3-haiku</AntSelect.Option>
            </AntSelect>
          </Form.Item>

          <Form.Item
            label="Allowed Region"
            name="allowed_model_region"
          >
            <AntSelect placeholder="Any region">
              <AntSelect.Option value="">Any region</AntSelect.Option>
              <AntSelect.Option value="us">US</AntSelect.Option>
              <AntSelect.Option value="eu">EU</AntSelect.Option>
            </AntSelect>
          </Form.Item>
        </div>

        <Form.Item
          label="Budget Duration"
          name="budget_duration"
        >
          <Input placeholder="e.g. 30d, 24h, 60m" />
        </Form.Item>

        <Form.Item
          label="Metadata (JSON)"
          name="metadata"
          initialValue="{}"
        >
          <Input.TextArea
            rows={3}
            className="font-mono"
            placeholder="{}"
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default CreateCustomerModal;
