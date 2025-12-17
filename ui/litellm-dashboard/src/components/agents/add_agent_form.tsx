import React, { useState, useEffect } from "react";
import { Modal, Form, message, Select, Input } from "antd";
import { Button } from "@tremor/react";
import { createAgentCall, getAgentCreateMetadata, AgentCreateInfo } from "../networking";
import AgentFormFields from "./agent_form_fields";
import DynamicAgentFormFields, { buildDynamicAgentData } from "./dynamic_agent_form_fields";
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
  const [agentType, setAgentType] = useState<string>("a2a");
  const [agentTypeMetadata, setAgentTypeMetadata] = useState<AgentCreateInfo[]>([]);
  const [loadingMetadata, setLoadingMetadata] = useState(false);

  // Fetch agent type metadata on mount
  useEffect(() => {
    const fetchMetadata = async () => {
      setLoadingMetadata(true);
      try {
        const metadata = await getAgentCreateMetadata();
        setAgentTypeMetadata(metadata);
      } catch (error) {
        console.error("Error fetching agent metadata:", error);
      } finally {
        setLoadingMetadata(false);
      }
    };
    fetchMetadata();
  }, []);

  const selectedAgentTypeInfo = agentTypeMetadata.find(
    (info) => info.agent_type === agentType
  );

  const handleSubmit = async (values: any) => {
    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    setIsSubmitting(true);
    try {
      let agentData: any;

      if (agentType === "a2a") {
        agentData = buildAgentDataFromForm(values);
      } else if (selectedAgentTypeInfo?.use_a2a_form_fields) {
        // A2A-compatible agents use the standard A2A form builder
        // but need to add litellm_params from the agent type config
        agentData = buildAgentDataFromForm(values);
        
        // Merge litellm_params_template
        if (selectedAgentTypeInfo.litellm_params_template) {
          agentData.litellm_params = {
            ...agentData.litellm_params,
            ...selectedAgentTypeInfo.litellm_params_template,
          };
        }
        
        // Add credential fields to litellm_params
        for (const field of selectedAgentTypeInfo.credential_fields) {
          const value = values[field.key];
          if (value && field.include_in_litellm_params !== false) {
            agentData.litellm_params[field.key] = value;
          }
        }
      } else if (selectedAgentTypeInfo) {
        agentData = buildDynamicAgentData(values, selectedAgentTypeInfo);
      }

      await createAgentCall(accessToken, agentData);
      message.success("Agent created successfully");
      form.resetFields();
      setAgentType("a2a");
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
    setAgentType("a2a");
    onClose();
  };

  const handleAgentTypeChange = (value: string) => {
    setAgentType(value);
    form.resetFields();
  };

  // Get the logo for the selected agent type for the header
  const selectedLogo = selectedAgentTypeInfo?.logo_url || agentTypeMetadata.find(a => a.agent_type === "a2a")?.logo_url;

  return (
    <Modal
      title={
        <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
          {selectedLogo && (
            <img
              src={selectedLogo}
              alt="Agent"
              className="w-6 h-6 object-contain"
            />
          )}
          <h2 className="text-xl font-semibold text-gray-900">Add New Agent</h2>
        </div>
      }
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={900}
      className="top-8"
      styles={{
        body: { padding: "24px" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
    >
      <div className="mt-4">
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={agentType === "a2a" ? getDefaultFormValues() : {}}
          className="space-y-4"
        >
          {/* Agent Type Selection */}
          <Form.Item
            label={<span className="text-sm font-medium text-gray-700">Agent Type</span>}
            required
            tooltip="Select the type of agent you want to create"
          >
            <Select
              value={agentType}
              onChange={handleAgentTypeChange}
              size="large"
              style={{ width: "100%" }}
              optionLabelProp="label"
            >
              {agentTypeMetadata.map((info) => (
                <Select.Option 
                  key={info.agent_type} 
                  value={info.agent_type}
                  label={
                    <div className="flex items-center gap-2">
                      <img src={info.logo_url || ""} alt="" className="w-4 h-4 object-contain" />
                      <span>{info.agent_type_display_name}</span>
                    </div>
                  }
                >
                  <div className="flex items-center gap-3 py-1">
                    <img
                      src={info.logo_url || ""}
                      alt={info.agent_type_display_name}
                      className="w-5 h-5 object-contain"
                    />
                    <div>
                      <div className="font-medium">{info.agent_type_display_name}</div>
                      {info.description && (
                        <div className="text-xs text-gray-500">{info.description}</div>
                      )}
                    </div>
                  </div>
                </Select.Option>
              ))}
            </Select>
          </Form.Item>

          {/* Conditional Form Fields */}
          <div className="mt-6">
            {agentType === "a2a" ? (
              <AgentFormFields showAgentName={true} />
            ) : selectedAgentTypeInfo?.use_a2a_form_fields ? (
              // A2A-compatible agents (like Pydantic AI) use full A2A form fields
              // plus any additional credential fields
              <>
                <AgentFormFields showAgentName={true} />
                {selectedAgentTypeInfo.credential_fields.length > 0 && (
                  <div className="mt-4 p-4 border border-gray-200 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-700 mb-3">
                      {selectedAgentTypeInfo.agent_type_display_name} Settings
                    </h4>
                    {selectedAgentTypeInfo.credential_fields.map((field) => (
                      <Form.Item
                        key={field.key}
                        label={field.label}
                        name={field.key}
                        rules={field.required ? [{ required: true, message: `Please enter ${field.label}` }] : undefined}
                        tooltip={field.tooltip}
                        initialValue={field.default_value}
                      >
                        {field.field_type === "password" ? (
                          <Input.Password placeholder={field.placeholder || ""} />
                        ) : (
                          <Input placeholder={field.placeholder || ""} />
                        )}
                      </Form.Item>
                    ))}
                  </div>
                )}
              </>
            ) : selectedAgentTypeInfo ? (
              <DynamicAgentFormFields agentTypeInfo={selectedAgentTypeInfo} />
            ) : null}
          </div>

          {/* Footer Buttons */}
          <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100 mt-6">
            <Button variant="secondary" onClick={handleCancel}>
              Cancel
            </Button>
            <Button variant="primary" loading={isSubmitting}>
              {isSubmitting ? "Creating..." : "Create Agent"}
            </Button>
          </div>
        </Form>
      </div>
    </Modal>
  );
};

export default AddAgentForm;
