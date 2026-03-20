import { InfoCircleOutlined } from "@ant-design/icons";
import { Accordion, AccordionBody, AccordionHeader, Button, TextInput, Title } from "@tremor/react";
import { Form, Input, Modal, Select as Select2, Tooltip } from "antd";
import React from "react";
import BudgetDurationDropdown from "../../common_components/budget_duration_dropdown";
import NumericalInput from "../../shared/numerical_input";

interface ModelInfo {
  model_name: string;
  litellm_params: {
    model: string;
  };
  model_info: {
    id: string;
  };
}

interface CreateTagModalProps {
  visible: boolean;
  onCancel: () => void;
  onSubmit: (values: any) => void;
  availableModels: ModelInfo[];
}

const CreateTagModal: React.FC<CreateTagModalProps> = ({ visible, onCancel, onSubmit, availableModels }) => {
  const [form] = Form.useForm();

  const handleFinish = (values: any) => {
    onSubmit(values);
    form.resetFields();
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  return (
    <Modal title="Create New Tag" open={visible} width={800} footer={null} onCancel={handleCancel}>
      <Form form={form} onFinish={handleFinish} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
        <Form.Item label="Tag Name" name="tag_name" rules={[{ required: true, message: "Please input a tag name" }]}>
          <TextInput />
        </Form.Item>

        <Form.Item label="Description" name="description">
          <Input.TextArea rows={4} />
        </Form.Item>

        <Form.Item
          label={
            <span>
              Allowed Models
              <Tooltip title="Select which models are allowed to process requests from this tag">
                <InfoCircleOutlined style={{ marginLeft: "4px" }} />
              </Tooltip>
            </span>
          }
          name="allowed_llms"
        >
          <Select2 mode="multiple" placeholder="Select Models">
            {availableModels.map((model) => (
              <Select2.Option key={model.model_info.id} value={model.model_info.id}>
                <div>
                  <span>{model.model_name}</span>
                  <span className="text-gray-400 ml-2">({model.model_info.id})</span>
                </div>
              </Select2.Option>
            ))}
          </Select2>
        </Form.Item>

        <Accordion className="mt-4 mb-4">
          <AccordionHeader>
            <Title className="m-0">Budget & Rate Limits (Optional)</Title>
          </AccordionHeader>
          <AccordionBody>
            <Form.Item
              className="mt-4"
              label={
                <span>
                  Max Budget (USD){" "}
                  <Tooltip title="Maximum amount in USD this tag can spend. When reached, requests with this tag will be blocked">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="max_budget"
            >
              <NumericalInput step={0.01} precision={2} width={200} />
            </Form.Item>
            <Form.Item
              className="mt-4"
              label={
                <span>
                  Reset Budget{" "}
                  <Tooltip title="How often the budget should reset. For example, setting 'daily' will reset the budget every 24 hours">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="budget_duration"
            >
              <BudgetDurationDropdown onChange={(value) => form.setFieldValue("budget_duration", value)} />
            </Form.Item>

            <div className="mt-4 p-3 bg-gray-50 rounded-md border border-gray-200">
              <p className="text-sm text-gray-600">
                TPM/RPM limits for tags are not currently supported. If you need this feature, please{" "}
                <a
                  href="https://github.com/BerriAI/litellm/issues/new"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 underline"
                >
                  create a GitHub issue
                </a>
                .
              </p>
            </div>
          </AccordionBody>
        </Accordion>

        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button type="submit">Create Tag</Button>
        </div>
      </Form>
    </Modal>
  );
};

export default CreateTagModal;
