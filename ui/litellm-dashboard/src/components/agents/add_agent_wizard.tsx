import React, { useState, useEffect } from "react";
import { Modal, Form, message, Select, Input, InputNumber, Tooltip, Switch } from "antd";
import { Button, TextInput } from "@tremor/react";
import { InfoCircleOutlined, CheckCircleOutlined, CopyOutlined, KeyOutlined } from "@ant-design/icons";
import {
  createAgentCall,
  getAgentCreateMetadata,
  AgentCreateInfo,
  keyCreateCall,
} from "../networking";
import { getDefaultFormValues, buildAgentDataFromForm } from "./agent_config";
import { buildDynamicAgentData } from "./dynamic_agent_form_fields";
import { CheckCircleIcon } from "@heroicons/react/outline";

const WIZARD_STEPS = [
  { key: 1, label: "Configure" },
  { key: 2, label: "Capabilities" },
  { key: 3, label: "Assign Key" },
  { key: 4, label: "Ready" },
];

const AVAILABLE_TOOLS = [
  { id: "web_search", name: "Web Search", description: "Search the internet for real-time info" },
  { id: "code_interpreter", name: "Code Interpreter", description: "Execute Python/JS code in a sandbox" },
  { id: "http_request", name: "HTTP Request", description: "Make outbound API calls" },
  { id: "file_reader", name: "File Reader", description: "Read and parse uploaded files" },
  { id: "database_query", name: "Database Query", description: "Run read-only SQL queries" },
];

const AVAILABLE_SKILLS = [
  { id: "summarization", name: "Summarization", description: "Condense long content into key points" },
  { id: "data_analysis", name: "Data Analysis", description: "Interpret and visualize datasets" },
  { id: "code_review", name: "Code Review", description: "Analyze code for bugs and improvements" },
  { id: "customer_support", name: "Customer Support", description: "Handle user queries and escalations" },
  { id: "tool_orchestration", name: "Tool Orchestration", description: "Chain multiple tools in sequence" },
  { id: "multi_step_reasoning", name: "Multi-step Reasoning", description: "Break down complex tasks into steps" },
];

interface AddAgentWizardProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  userID?: string;
  onSuccess: () => void;
}

interface CreatedAgentResult {
  agent_id: string;
  agent_name: string;
  agent_type_display?: string;
  url?: string;
  max_iterations?: number;
  tools_count?: number;
  skills_count?: number;
  key_name?: string;
  key_value?: string;
}

const StepIndicator: React.FC<{ currentStep: number }> = ({ currentStep }) => (
  <div className="flex items-center justify-center mb-8">
    {WIZARD_STEPS.map((step, idx) => (
      <React.Fragment key={step.key}>
        <div className="flex flex-col items-center">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold transition-colors ${
              step.key < currentStep
                ? "bg-blue-600 text-white"
                : step.key === currentStep
                ? "bg-blue-600 text-white"
                : "bg-gray-200 text-gray-500"
            }`}
          >
            {step.key < currentStep ? (
              <CheckCircleIcon className="w-5 h-5" />
            ) : (
              step.key
            )}
          </div>
          <span
            className={`text-xs mt-1 ${
              step.key === currentStep
                ? "text-blue-600 font-semibold"
                : "text-gray-400"
            }`}
          >
            {step.label}
          </span>
        </div>
        {idx < WIZARD_STEPS.length - 1 && (
          <div
            className={`w-16 h-0.5 mx-2 mb-5 ${
              step.key < currentStep ? "bg-blue-600" : "bg-gray-200"
            }`}
          />
        )}
      </React.Fragment>
    ))}
  </div>
);

const ToolCard: React.FC<{
  tool: typeof AVAILABLE_TOOLS[0];
  selected: boolean;
  onToggle: () => void;
}> = ({ tool, selected, onToggle }) => (
  <div
    onClick={onToggle}
    className={`flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-all ${
      selected
        ? "border-blue-500 bg-blue-50"
        : "border-gray-200 hover:border-gray-300 bg-white"
    }`}
  >
    <div>
      <div className="text-sm font-medium text-gray-900">{tool.name}</div>
      <div className="text-xs text-gray-500">{tool.description}</div>
    </div>
    <Switch checked={selected} size="small" onClick={(_, e) => e.stopPropagation()} onChange={onToggle} />
  </div>
);

const SkillCard: React.FC<{
  skill: typeof AVAILABLE_SKILLS[0];
  selected: boolean;
  onToggle: () => void;
}> = ({ skill, selected, onToggle }) => (
  <div
    onClick={onToggle}
    className={`p-3 rounded-lg border cursor-pointer transition-all ${
      selected
        ? "border-blue-500 bg-blue-50"
        : "border-gray-200 hover:border-gray-300 bg-white"
    }`}
  >
    <div className="flex items-center justify-between">
      <span className="text-sm font-medium text-gray-900">{skill.name}</span>
      <Switch checked={selected} size="small" onClick={(_, e) => e.stopPropagation()} onChange={onToggle} />
    </div>
    <div className="text-xs text-gray-500 mt-1">{skill.description}</div>
  </div>
);

const ITERATION_PRESETS = [5, 10, 25, 50];

const AddAgentWizard: React.FC<AddAgentWizardProps> = ({
  visible,
  onClose,
  accessToken,
  userID,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [agentType, setAgentType] = useState<string>("a2a");
  const [agentTypeMetadata, setAgentTypeMetadata] = useState<AgentCreateInfo[]>([]);

  const [maxIterations, setMaxIterations] = useState(10);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);

  const [keyOption, setKeyOption] = useState<"create" | "existing" | "skip">("create");
  const [keyName, setKeyName] = useState("new-agent-key");

  const [createdAgent, setCreatedAgent] = useState<CreatedAgentResult | null>(null);

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const metadata = await getAgentCreateMetadata();
        setAgentTypeMetadata(metadata);
      } catch (error) {
        console.error("Error fetching agent metadata:", error);
      }
    };
    fetchMetadata();
  }, []);

  const selectedAgentTypeInfo = agentTypeMetadata.find(
    (info) => info.agent_type === agentType
  );

  const resetWizard = () => {
    setCurrentStep(1);
    form.resetFields();
    setAgentType("a2a");
    setMaxIterations(10);
    setSelectedTools([]);
    setSelectedSkills([]);
    setKeyOption("create");
    setKeyName("new-agent-key");
    setCreatedAgent(null);
  };

  const handleClose = () => {
    resetWizard();
    onClose();
  };

  const handleAgentTypeChange = (value: string) => {
    setAgentType(value);
    form.resetFields();
  };

  const toggleTool = (toolId: string) => {
    setSelectedTools((prev) =>
      prev.includes(toolId) ? prev.filter((t) => t !== toolId) : [...prev, toolId]
    );
  };

  const toggleSkill = (skillId: string) => {
    setSelectedSkills((prev) =>
      prev.includes(skillId) ? prev.filter((s) => s !== skillId) : [...prev, skillId]
    );
  };

  const handleNext = async () => {
    if (currentStep === 1) {
      try {
        await form.validateFields();
        setCurrentStep(2);
      } catch {
        return;
      }
    } else if (currentStep === 2) {
      setCurrentStep(3);
    } else if (currentStep === 3) {
      await handleCreateAgent();
    }
  };

  const handleBack = () => {
    if (currentStep > 1 && currentStep < 4) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleCreateAgent = async () => {
    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    setIsSubmitting(true);
    try {
      const values = form.getFieldsValue(true);

      let agentData: any;
      if (agentType === "a2a") {
        agentData = buildAgentDataFromForm(values);
      } else if (selectedAgentTypeInfo?.use_a2a_form_fields) {
        agentData = buildAgentDataFromForm(values);
        if (selectedAgentTypeInfo.litellm_params_template) {
          agentData.litellm_params = {
            ...agentData.litellm_params,
            ...selectedAgentTypeInfo.litellm_params_template,
          };
        }
        for (const field of selectedAgentTypeInfo.credential_fields) {
          const value = values[field.key];
          if (value && field.include_in_litellm_params !== false) {
            agentData.litellm_params = agentData.litellm_params || {};
            agentData.litellm_params[field.key] = value;
          }
        }
      } else if (selectedAgentTypeInfo) {
        agentData = buildDynamicAgentData(values, selectedAgentTypeInfo);
      }

      if (!agentData.litellm_params) {
        agentData.litellm_params = {};
      }
      agentData.litellm_params.max_iterations = maxIterations;
      agentData.litellm_params.tools = selectedTools;

      const existingSkills = agentData.agent_card_params?.skills || [];
      const wizardSkills = selectedSkills.map((skillId) => {
        const skillDef = AVAILABLE_SKILLS.find((s) => s.id === skillId);
        return {
          id: skillId,
          name: skillDef?.name || skillId,
          description: skillDef?.description || "",
          tags: [skillId],
        };
      });
      if (agentData.agent_card_params) {
        agentData.agent_card_params.skills = [...existingSkills, ...wizardSkills];
      }

      const response = await createAgentCall(accessToken, agentData);

      let createdKeyName: string | undefined;
      let createdKeyValue: string | undefined;
      if (keyOption === "create" && keyName.trim()) {
        try {
          const keyResp = await keyCreateCall(accessToken, userID || "", {
            key_alias: keyName.trim(),
          });
          createdKeyName = keyName.trim();
          createdKeyValue = keyResp?.key;
        } catch (keyError) {
          console.error("Error creating key:", keyError);
        }
      }

      const agentName = values.agent_name || values.name || "new-agent";
      const displayType =
        selectedAgentTypeInfo?.agent_type_display_name || "A2A Standard";

      setCreatedAgent({
        agent_id: response?.agent_id || "",
        agent_name: agentName,
        agent_type_display: displayType,
        url: values.url || "",
        max_iterations: maxIterations,
        tools_count: selectedTools.length,
        skills_count: selectedSkills.length,
        key_name: createdKeyName,
        key_value: createdKeyValue,
      });

      setCurrentStep(4);
      onSuccess();
    } catch (error) {
      console.error("Error creating agent:", error);
      message.error("Failed to create agent");
    } finally {
      setIsSubmitting(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    message.success("Copied to clipboard");
  };

  const renderStep1 = () => (
    <div className="grid grid-cols-1 gap-6">
      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Agent Type
            <Tooltip title="Select the type of agent you want to create">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        required
      >
        <Select
          value={agentType}
          onChange={handleAgentTypeChange}
          size="large"
          className="rounded-lg"
          style={{ width: "100%" }}
          optionLabelProp="label"
        >
          {agentTypeMetadata.map((info) => (
            <Select.Option
              key={info.agent_type}
              value={info.agent_type}
              label={
                <div className="flex items-center gap-2">
                  {info.logo_url && (
                    <img src={info.logo_url} alt="" className="w-4 h-4 object-contain" />
                  )}
                  <span>{info.agent_type_display_name}</span>
                </div>
              }
            >
              <div className="flex items-center gap-3 py-1">
                {info.logo_url && (
                  <img src={info.logo_url} alt={info.agent_type_display_name} className="w-5 h-5 object-contain" />
                )}
                <div>
                  <div className="font-medium">{info.agent_type_display_name}</div>
                  {info.description && <div className="text-xs text-gray-500">{info.description}</div>}
                </div>
              </div>
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Agent Name
            <Tooltip title="Unique identifier for the agent. Cannot contain spaces.">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        name="agent_name"
        rules={[{ required: true, message: "Please enter a unique agent name" }]}
      >
        <TextInput
          placeholder="e.g., customer-support-agent"
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>

      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">Display Name</span>}
        name="name"
        rules={[{ required: true, message: "Please enter a display name" }]}
      >
        <TextInput
          placeholder="e.g., Customer Support Agent"
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>

      <Form.Item
        label={<span className="text-sm font-medium text-gray-700">Description</span>}
        name="description"
        rules={[{ required: true, message: "Please enter a description" }]}
      >
        <TextInput
          placeholder="Brief description of what this agent does"
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>

      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center">
            URL
            <Tooltip title="Base URL where the agent is hosted">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        name="url"
        rules={[{ required: true, message: "Please enter the agent URL" }]}
      >
        <TextInput
          placeholder="http://localhost:9999/"
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
      </Form.Item>
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-8">
      {/* Max Iterations */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Max Iterations
            <Tooltip title="How many reasoning loops before the agent stops. Lower values reduce cost; higher values allow more complex tasks.">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
          <span className="text-xs text-gray-500">Range: 1–100</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center border border-gray-300 rounded-lg overflow-hidden">
            <button
              type="button"
              className="px-3 py-2 text-gray-500 hover:bg-gray-100 border-r border-gray-300"
              onClick={() => setMaxIterations(Math.max(1, maxIterations - 1))}
            >
              &minus;
            </button>
            <InputNumber
              min={1}
              max={100}
              value={maxIterations}
              onChange={(val) => setMaxIterations(val || 10)}
              controls={false}
              className="w-16 text-center border-none"
              style={{ textAlign: "center" }}
            />
            <button
              type="button"
              className="px-3 py-2 text-gray-500 hover:bg-gray-100 border-l border-gray-300"
              onClick={() => setMaxIterations(Math.min(100, maxIterations + 1))}
            >
              +
            </button>
          </div>
          {ITERATION_PRESETS.map((preset) => (
            <button
              key={preset}
              type="button"
              onClick={() => setMaxIterations(preset)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                maxIterations === preset
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {preset}
            </button>
          ))}
        </div>
      </div>

      {/* Tools */}
      <div className="pt-6 border-t border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Tools
            <Tooltip title="Select which tools this agent is allowed to use during execution.">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
          <span className="text-xs text-gray-500">{selectedTools.length} selected</span>
        </div>
        <div className="space-y-2">
          {AVAILABLE_TOOLS.map((tool) => (
            <ToolCard
              key={tool.id}
              tool={tool}
              selected={selectedTools.includes(tool.id)}
              onToggle={() => toggleTool(tool.id)}
            />
          ))}
        </div>
      </div>

      {/* Skills */}
      <div className="pt-6 border-t border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Skills
            <Tooltip title="Select which skills this agent should have. Skills describe the agent's advertised capabilities.">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
          <span className="text-xs text-gray-500">{selectedSkills.length} selected</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {AVAILABLE_SKILLS.map((skill) => (
            <SkillCard
              key={skill.id}
              skill={skill}
              selected={selectedSkills.includes(skill.id)}
              onToggle={() => toggleSkill(skill.id)}
            />
          ))}
        </div>
      </div>
    </div>
  );

  const renderStep3 = () => {
    const agentName = form.getFieldValue("agent_name") || "your-agent";
    return (
      <div className="space-y-6">
        <div className="text-center mb-4">
          <span className="inline-flex items-center gap-2 px-2 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded-md text-sm font-medium">
            {agentName}
          </span>
        </div>

        {/* Create new key */}
        <div
          onClick={() => setKeyOption("create")}
          className={`p-5 rounded-lg border cursor-pointer transition-all ${
            keyOption === "create" ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:border-gray-300"
          }`}
        >
          <div className="flex items-start gap-3">
            <div
              className={`w-5 h-5 rounded-full border-2 mt-0.5 flex items-center justify-center flex-shrink-0 ${
                keyOption === "create" ? "border-blue-600" : "border-gray-300"
              }`}
            >
              {keyOption === "create" && <div className="w-2.5 h-2.5 rounded-full bg-blue-600" />}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <KeyOutlined className="text-gray-500" />
                <span className="text-sm font-semibold text-gray-900">Create a new key for this agent</span>
                <span className="text-xs bg-green-50 text-green-700 border border-green-200 px-2 py-0.5 rounded-md font-medium">
                  Recommended
                </span>
              </div>
              <p className="text-xs text-gray-500 ml-6">
                A dedicated key scoped to this agent. Recommended for production.
              </p>
              {keyOption === "create" && (
                <div className="mt-4 ml-6 space-y-3">
                  <div>
                    <label className="text-xs text-gray-600 font-medium">Key Name</label>
                    <TextInput
                      value={keyName}
                      onChange={(e) => setKeyName(e.target.value)}
                      placeholder="new-agent-key"
                      className="mt-1 rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-600 font-medium">Models</label>
                    <Input value="All models (default)" disabled className="mt-1 rounded-lg" />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Assign existing key */}
        <div
          onClick={() => setKeyOption("existing")}
          className={`p-5 rounded-lg border cursor-pointer transition-all ${
            keyOption === "existing" ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:border-gray-300"
          }`}
        >
          <div className="flex items-start gap-3">
            <div
              className={`w-5 h-5 rounded-full border-2 mt-0.5 flex items-center justify-center flex-shrink-0 ${
                keyOption === "existing" ? "border-blue-600" : "border-gray-300"
              }`}
            >
              {keyOption === "existing" && <div className="w-2.5 h-2.5 rounded-full bg-blue-600" />}
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <KeyOutlined className="text-gray-500" />
                <span className="text-sm font-semibold text-gray-900">Assign an existing key</span>
              </div>
              <p className="text-xs text-gray-500 ml-6">
                Link a key you&apos;ve already created.
              </p>
            </div>
          </div>
        </div>

        {/* Skip */}
        <div className="text-center">
          <button
            type="button"
            onClick={() => setKeyOption("skip")}
            className={`text-sm underline bg-transparent border-none cursor-pointer ${
              keyOption === "skip" ? "text-blue-600" : "text-gray-400 hover:text-gray-600"
            }`}
          >
            Skip for now — I&apos;ll assign a key later
          </button>
        </div>
      </div>
    );
  };

  const renderStep4 = () => {
    if (!createdAgent) return null;

    return (
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <div className="w-14 h-14 rounded-full bg-green-50 border border-green-200 flex items-center justify-center">
            <CheckCircleOutlined className="text-2xl text-green-600" />
          </div>
        </div>

        <div>
          <h3 className="text-xl font-semibold text-gray-900">Agent created successfully!</h3>
          <p className="text-sm text-gray-500 mt-1">Your agent is configured and ready to use.</p>
        </div>

        <div className="flex gap-4 justify-center text-left">
          {/* Agent details */}
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-5 flex-1 max-w-[280px]">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Agent Details</p>
            <div className="space-y-2">
              <p className="text-sm font-semibold text-gray-900">{createdAgent.agent_name}</p>
              {createdAgent.agent_type_display && (
                <p className="text-xs text-gray-500 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-gray-400" />
                  {createdAgent.agent_type_display}
                </p>
              )}
              {createdAgent.url && (
                <p className="text-xs text-gray-500">{createdAgent.url}</p>
              )}
              <p className="text-xs text-gray-500">Max {createdAgent.max_iterations} iterations</p>
              <p className="text-xs text-gray-500">
                {createdAgent.tools_count} tools, {createdAgent.skills_count} skills
              </p>
            </div>
          </div>

          {/* Key info */}
          {createdAgent.key_name && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-5 flex-1 max-w-[280px]">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">API Key Assigned</p>
              <p className="text-sm font-semibold text-gray-900">{createdAgent.key_name}</p>
              {createdAgent.key_value && (
                <div className="mt-2 flex items-center gap-2 bg-white rounded-md border border-gray-200 px-3 py-2">
                  <code className="text-xs text-gray-600 truncate flex-1 font-mono">
                    {createdAgent.key_value.substring(0, 12)}..
                    {createdAgent.key_value.substring(createdAgent.key_value.length - 4)}
                  </code>
                  <Tooltip title="Copy to clipboard">
                    <CopyOutlined
                      onClick={() => copyToClipboard(createdAgent.key_value!)}
                      className="cursor-pointer text-gray-400 hover:text-blue-500"
                    />
                  </Tooltip>
                </div>
              )}
              <p className="text-xs text-green-600 mt-2 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                Active &amp; ready
              </p>
            </div>
          )}
        </div>

        <div className="flex gap-3 justify-center pt-2">
          <Button variant="secondary" onClick={handleClose}>
            View All Agents
          </Button>
          <Button variant="primary" onClick={handleClose}>
            Go to Virtual Keys
          </Button>
        </div>
      </div>
    );
  };

  const getStepContent = () => {
    switch (currentStep) {
      case 1: return renderStep1();
      case 2: return renderStep2();
      case 3: return renderStep3();
      case 4: return renderStep4();
      default: return null;
    }
  };

  return (
    <Modal
      title={
        <div className="flex items-center pb-4 border-b border-gray-100" style={{ gap: 12 }}>
          <h2 className="text-xl font-semibold text-gray-900">Add New Agent</h2>
        </div>
      }
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={900}
      className="top-8"
      styles={{
        body: { padding: "24px", maxHeight: "70vh", overflowY: "auto" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
      destroyOnClose
    >
      <div className="mt-6">
        <StepIndicator currentStep={currentStep} />

        <Form
          form={form}
          layout="vertical"
          initialValues={getDefaultFormValues()}
          className="space-y-6"
        >
          {getStepContent()}
        </Form>

        {currentStep < 4 && (
          <div className="flex items-center justify-between pt-6 border-t border-gray-100 mt-6">
            <div>
              {currentStep > 1 && (
                <Button variant="light" onClick={handleBack}>
                  Back
                </Button>
              )}
            </div>
            <Button variant="primary" onClick={handleNext} loading={isSubmitting}>
              {currentStep === 3
                ? isSubmitting ? "Creating..." : "Create Agent"
                : "Next"}
            </Button>
          </div>
        )}
      </div>
    </Modal>
  );
};

export default AddAgentWizard;
