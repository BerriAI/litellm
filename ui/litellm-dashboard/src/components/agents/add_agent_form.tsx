import React, { useState } from "react";
import { Modal, Form, Button as AntButton, message } from "antd";
import { createAgentCall } from "../networking";
import AgentFormFields from "./agent_form_fields";
import { getDefaultFormValues, buildAgentDataFromForm } from "./agent_config";

interface AddAgentFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

const AddAgentForm: React.FC<AddAgentFormProps> = ({
  visible,
  onClose,
  accessToken,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (values: any) => {
    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    setIsSubmitting(true);
    try {
      const agentData = buildAgentDataFromForm(values);
      await createAgentCall(accessToken, agentData);
      message.success("Agent created successfully");
      form.resetFields();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error creating agent:", error);
      message.error("Failed to create agent");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onClose();
  };

  return (
    <Modal
      title="Add New Agent"
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={800}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={getDefaultFormValues()}
      >
        <AgentFormFields showAgentName={true} />

        <Form.Item>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px" }}>
            <AntButton onClick={handleCancel}>
              Cancel
            </AntButton>
            <AntButton
              htmlType="submit"
              loading={isSubmitting}
            >
              Create Agent
            </AntButton>
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddAgentForm;

