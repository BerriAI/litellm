import {
  Button as Button2,
  Modal,
  Form,
  Select as Select2,
  InputNumber,
  message,
} from "antd";

import {
  TextInput,
  Button,
} from "@tremor/react";

import { organizationCreateCall } from "../networking";

// types.ts
export interface FormData {
  name: string;
  models: string[];
  maxBudget: number | null;
  budgetDuration: string | null;
  tpmLimit: number | null;
  rpmLimit: number | null;
}
  
export interface OrganizationFormProps {
  title?: string;
  onCancel?: () => void;
  accessToken: string | null;
  availableModels?: string[];
  initialValues?: Partial<FormData>;
  submitButtonText?: string;
  modelSelectionType?: 'single' | 'multiple';
}

// OrganizationForm.tsx
import React, { useState } from 'react';

const onSubmit = async (formValues: Record<string, any>, accessToken: string | null, setIsModalVisible: any) => {
  if (accessToken == null) {
    return;
  }
  try {
    message.info("Creating Organization");
    console.log("formValues: " + JSON.stringify(formValues));
    const response: any = await organizationCreateCall(accessToken, formValues);
    console.log(`response for organization create call: ${response}`);
    message.success("Organization created");
    sessionStorage.removeItem('organizations');
    setIsModalVisible(false);
  } catch (error) {
    console.error("Error creating the organization:", error);
    message.error("Error creating the organization: " + error, 20);
  }

}

const OrganizationForm: React.FC<OrganizationFormProps> = ({
  title = "Create Organization",
  onCancel,
  accessToken,
  availableModels = [],
  initialValues = {},
  submitButtonText = "Create",
  modelSelectionType = "multiple",
}) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState<boolean>(false);
  const [formData, setFormData] = useState<FormData>({
    name: initialValues.name || '',
    models: initialValues.models || [],
    maxBudget: initialValues.maxBudget || null,
    budgetDuration: initialValues.budgetDuration || null,
    tpmLimit: initialValues.tpmLimit || null,
    rpmLimit: initialValues.rpmLimit || null
  });

  console.log(`availableModels: ${availableModels}`)

  const handleSubmit = async (formValues: Record<string, any>) => {
    if (accessToken == null) {
      return;
    }
    await onSubmit(formValues, accessToken, setIsModalVisible);
    setIsModalVisible(false);
  };

  const handleCancel = (): void => {
    setIsModalVisible(false);
    if (onCancel) onCancel();
  };

  return (
    <div className="w-full">
      <Button
        onClick={() => setIsModalVisible(true)}
        className="mx-auto"
        type="button"
      >
        + Create New {title}
      </Button>

      <Modal
            title={`Create ${title}`}
            visible={isModalVisible}
            width={800}
            footer={null}
            onCancel={handleCancel}
          >
            <Form
              form={form}
              onFinish={handleSubmit}
              labelCol={{ span: 8 }}
              wrapperCol={{ span: 16 }}
              labelAlign="left"
            >
              <>
                <Form.Item
                  label={`${title} Name`}
                  name="organization_alias"
                  rules={[
                    { required: true, message: `Please input a ${title} name` },
                  ]}
                >
                  <TextInput placeholder="" />
                </Form.Item>
                <Form.Item label="Models" name="models">
                  <Select2
                    mode="multiple"
                    placeholder="Select models"
                    style={{ width: "100%" }}
                  >
                    <Select2.Option
                      key="all-proxy-models"
                      value="all-proxy-models"
                    >
                      All Proxy Models
                    </Select2.Option>
                    {availableModels.map((model) => (
                      <Select2.Option key={model} value={model}>
                        {model}
                      </Select2.Option>
                    ))}
                  </Select2>
                </Form.Item>

                <Form.Item label="Max Budget (USD)" name="max_budget">
                  <InputNumber step={0.01} precision={2} width={200} />
                </Form.Item>
                <Form.Item
                  className="mt-8"
                  label="Reset Budget"
                  name="budget_duration"
                >
                  <Select2 defaultValue={null} placeholder="n/a">
                    <Select2.Option value="24h">daily</Select2.Option>
                    <Select2.Option value="7d">weekly</Select2.Option>
                    <Select2.Option value="30d">monthly</Select2.Option>
                  </Select2>
                </Form.Item>
                <Form.Item
                  label="Tokens per minute Limit (TPM)"
                  name="tpm_limit"
                >
                  <InputNumber step={1} width={400} />
                </Form.Item>
                <Form.Item
                  label="Requests per minute Limit (RPM)"
                  name="rpm_limit"
                >
                  <InputNumber step={1} width={400} />
                </Form.Item>
              </>
              <div style={{ textAlign: "right", marginTop: "10px" }}>
                <Button2 htmlType="submit">{submitButtonText}</Button2>
              </div>
            </Form>
          </Modal>
    </div>
  );
};

export default OrganizationForm;