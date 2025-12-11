import React, { useState, useEffect } from "react";
import { Modal, Form, Button as AntButton, message } from "antd";
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

// Local logo paths for agent types
const AGENT_TYPE_LOGOS: Record<string, string> = {
  langgraph: "/assets/logos/langgraph.png",
  a2a: "/assets/logos/a2a_agent.png",
};

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

  const getLogoForAgentType = (agentTypeKey: string, fallbackUrl?: string | null): string => {
    return AGENT_TYPE_LOGOS[agentTypeKey] || fallbackUrl || AGENT_TYPE_LOGOS.a2a;
  };

  return (
    <Modal
      title="Add New Agent"
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={agentType === "a2a" ? getDefaultFormValues() : {}}
      >
        {/* Agent Type Selection */}
        <Form.Item
          label="Agent Type"
          required
          tooltip="Select the type of agent you want to create"
        >
          <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
            {/* A2A Standard Option */}
            <div
              onClick={() => handleAgentTypeChange("a2a")}
              style={{
                border: agentType === "a2a" ? "2px solid #1890ff" : "1px solid #d9d9d9",
                borderRadius: "8px",
                padding: "16px 20px",
                cursor: "pointer",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                minWidth: "130px",
                minHeight: "110px",
                backgroundColor: agentType === "a2a" ? "#e6f7ff" : "white",
                transition: "all 0.2s ease",
              }}
            >
              <img
                src={AGENT_TYPE_LOGOS.a2a}
                alt="A2A"
                style={{ width: "40px", height: "40px", marginBottom: "10px", objectFit: "contain" }}
              />
              <span style={{ fontWeight: agentType === "a2a" ? 600 : 400, fontSize: "14px" }}>
                A2A Standard
              </span>
              <span style={{ fontSize: "11px", color: "#666", textAlign: "center", marginTop: "4px" }}>
                Standard A2A protocol
              </span>
            </div>

            {/* Dynamic Agent Types from API */}
            {agentTypeMetadata.map((info) => (
              <div
                key={info.agent_type}
                onClick={() => handleAgentTypeChange(info.agent_type)}
                style={{
                  border: agentType === info.agent_type ? "2px solid #1890ff" : "1px solid #d9d9d9",
                  borderRadius: "8px",
                  padding: "16px 20px",
                  cursor: "pointer",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  minWidth: "130px",
                  minHeight: "110px",
                  backgroundColor: agentType === info.agent_type ? "#e6f7ff" : "white",
                  transition: "all 0.2s ease",
                }}
              >
                <img
                  src={getLogoForAgentType(info.agent_type, info.logo_url)}
                  alt={info.agent_type_display_name}
                  style={{ width: "40px", height: "40px", marginBottom: "10px", objectFit: "contain" }}
                />
                <span style={{ fontWeight: agentType === info.agent_type ? 600 : 400, fontSize: "14px" }}>
                  {info.agent_type_display_name}
                </span>
                {info.description && (
                  <span style={{ fontSize: "11px", color: "#666", textAlign: "center", marginTop: "4px" }}>
                    {info.description.length > 35 ? `${info.description.slice(0, 35)}...` : info.description}
                  </span>
                )}
              </div>
            ))}
          </div>
        </Form.Item>

        {/* Conditional Form Fields */}
        {agentType === "a2a" ? (
          <AgentFormFields showAgentName={true} />
        ) : selectedAgentTypeInfo ? (
          <DynamicAgentFormFields agentTypeInfo={selectedAgentTypeInfo} />
        ) : null}

        <Form.Item style={{ marginBottom: 0, marginTop: "24px" }}>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px" }}>
            <AntButton onClick={handleCancel}>
              Cancel
            </AntButton>
            <AntButton
              htmlType="submit"
              loading={isSubmitting}
              type="primary"
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
