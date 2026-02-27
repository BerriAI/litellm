import React, { useEffect, useState } from "react";
import { Modal, Form, Input, InputNumber, Select as AntSelect, Switch } from "antd";
import type { Customer } from "@/app/(dashboard)/customers/types";

interface CustomerInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  customer: Customer | null;
  onSave: (customer: Customer) => void;
}

const CustomerInfoModal: React.FC<CustomerInfoModalProps> = ({
  isOpen,
  onClose,
  customer,
  onSave,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (customer) {
      form.setFieldsValue({
        ...customer,
        max_budget: customer.litellm_budget_table?.max_budget,
        budget_duration: customer.litellm_budget_table?.budget_duration,
      });
    }
  }, [customer, form]);

  const handleOk = async () => {
    if (!customer) return;

    try {
      const values = await form.validateFields();
      setLoading(true);
      onSave({
        ...customer,
        ...values,
      });
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

  if (!customer) return null;

  return (
    <Modal
      title="Customer Details"
      open={isOpen}
      onOk={handleOk}
      onCancel={handleCancel}
      confirmLoading={loading}
      okText="Save Changes"
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        className="mt-4"
      >
        <Form.Item label="Customer ID">
          <Input
            value={customer.user_id}
            disabled
            className="font-mono bg-gray-50"
          />
        </Form.Item>

        <Form.Item
          label="Alias"
          name="alias"
        >
          <Input placeholder="Customer alias" />
        </Form.Item>

        <div className="grid grid-cols-2 gap-4">
          <Form.Item label="Spend (USD)">
            <Input
              value={customer.spend.toFixed(4)}
              disabled
              className="bg-gray-50"
            />
          </Form.Item>

          <Form.Item
            label="Max Budget"
            name="max_budget"
          >
            <InputNumber
              className="w-full"
              placeholder="No limit"
              min={0}
            />
          </Form.Item>
        </div>

        <Form.Item
          label="Budget ID"
          name="budget_id"
        >
          <Input placeholder="e.g. free_tier" />
        </Form.Item>

        <div className="grid grid-cols-2 gap-4">
          <Form.Item
            label="Default Model"
            name="default_model"
          >
            <AntSelect placeholder="None">
              <AntSelect.Option value="">None</AntSelect.Option>
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
          <Input placeholder="e.g. 30d, 24h" />
        </Form.Item>

        <Form.Item
          label="Blocked"
          name="blocked"
          valuePropName="checked"
        >
          <div className="flex items-center gap-3">
            <Switch />
            <span className="text-sm text-gray-500">
              {form.getFieldValue("blocked")
                ? "This customer is currently blocked from making requests"
                : "This customer can make requests normally"}
            </span>
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default CustomerInfoModal;
