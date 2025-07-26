import React, { useState, useEffect } from "react"
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Modal, message } from "antd"
import { Row, Col } from "antd"
import { Text, Switch } from "@tremor/react"
import LiteLLMModelNameField from "./litellm_model_name"
import ConditionalPublicModelName from "./conditional_public_model_name"
import ProviderSpecificFields from "./provider_specific_fields"
import AdvancedSettings from "./advanced_settings"
import { Providers, providerLogoMap, getPlaceholder } from "../provider_info_helpers"
import type { Team } from "../key_team_helpers/key_list"
import { CredentialItem, modelAvailableCall } from "../networking"
import ConnectionErrorDisplay from "./model_connection_test"
import { TEST_MODES } from "./add_model_modes"
import TeamDropdown from "../common_components/team_dropdown"
import { all_admin_roles } from "@/utils/roles"
import { handleAddModelSubmit } from "./handle_add_model_submit"
import type { UploadProps } from "antd/es/upload"

interface AddModelFormProps {
  onSuccess: () => void
  onError: (error: string) => void
  selectedProvider: Providers
  setSelectedProvider: (provider: Providers) => void
  providerModels: string[]
  setProviderModelsFn: (provider: Providers) => void
  teams: Team[] | null
  credentials: CredentialItem[]
  accessToken: string
  userRole: string
  premiumUser: boolean
  uploadProps: UploadProps
}

const { Link } = Typography

const AddModelForm: React.FC<AddModelFormProps> = ({
  onSuccess,
  onError,
  selectedProvider,
  setSelectedProvider,
  providerModels,
  setProviderModelsFn,
  teams,
  credentials,
  accessToken,
  userRole,
  premiumUser,
  uploadProps,
}) => {
  // Create form instance for this component only
  const [form] = Form.useForm()

  // Local state
  const [testMode, setTestMode] = useState<string>("chat")
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false)
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false)
  const [connectionTestId, setConnectionTestId] = useState<string>("")
  const [isTeamOnly, setIsTeamOnly] = useState<boolean>(false)
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false)
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([])

  // Fetch model access groups
  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      try {
        const response = await modelAvailableCall(accessToken, "", "", false, null, true, true)
        setModelAccessGroups(response["data"].map((model: any) => model["id"]))
      } catch (error) {
        console.error("Failed to fetch model access groups:", error)
      }
    }
    fetchModelAccessGroups()
  }, [accessToken])

  const isAdmin = all_admin_roles.includes(userRole)

  // Handle form submission
  const handleSubmit = async (values: any) => {
    try {
      console.log("=== ADD MODEL FORM SUBMISSION ===")
      console.log("Form values:", values)

      await handleAddModelSubmit(values, accessToken, form, onSuccess)
      message.success("Model added successfully!")
      form.resetFields()
    } catch (error) {
      console.error("Add Model submission failed:", error)
      const errorMessage = error instanceof Error ? error.message : "Failed to add model"
      message.error(errorMessage)
      onError(errorMessage)
    }
  }

  // Handle test connection
  const handleTestConnection = async () => {
    setIsTestingConnection(true)
    setConnectionTestId(`test-${Date.now()}`)
    setIsResultModalVisible(true)
  }

  return (
    <>
      <Card>
        <Form
          form={form}
          onFinish={handleSubmit}
          labelCol={{ span: 10 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
          onFinishFailed={(errorInfo) => {
            console.log("Form validation failed:", errorInfo)
            if (errorInfo.errorFields && errorInfo.errorFields.length > 0) {
              const missingFields = errorInfo.errorFields.map((field: any) => {
                const fieldName = field.name[0]
                const friendlyNames: { [key: string]: string } = {
                  custom_llm_provider: "Provider",
                  model: "LiteLLM Model Name",
                  custom_model_name: "Custom Model Name",
                  api_key: "API Key",
                }
                return friendlyNames[fieldName] || fieldName
              })
              message.error(`Please fill in the following required fields: ${missingFields.join(", ")}`)
            }
          }}
        >
          {/* Provider Selection */}
          <Form.Item
            rules={[{ required: true, message: "Required" }]}
            label="Provider:"
            name="custom_llm_provider"
            tooltip="E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc."
          >
            <AntdSelect
              showSearch={true}
              value={selectedProvider}
              onChange={(value) => {
                setSelectedProvider(value)
                setProviderModelsFn(value)
                form.setFieldsValue({
                  model: [],
                  model_name: undefined,
                })
              }}
            >
              {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => (
                <AntdSelect.Option key={providerEnum} value={providerEnum}>
                  <div className="flex items-center space-x-2">
                    <img
                      src={providerLogoMap[providerDisplayName]}
                      alt={`${providerEnum} logo`}
                      className="w-5 h-5"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement
                        const parent = target.parentElement
                        if (parent) {
                          const fallbackDiv = document.createElement("div")
                          fallbackDiv.className =
                            "w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs"
                          fallbackDiv.textContent = providerDisplayName.charAt(0)
                          parent.replaceChild(fallbackDiv, target)
                        }
                      }}
                    />
                    <span>{providerDisplayName}</span>
                  </div>
                </AntdSelect.Option>
              ))}
            </AntdSelect>
          </Form.Item>

          {/* Model Name Field */}
          <LiteLLMModelNameField
            selectedProvider={selectedProvider}
            providerModels={providerModels}
            getPlaceholder={getPlaceholder}
            form={form}
          />

          {/* Public Model Name */}
          <ConditionalPublicModelName />

          {/* Mode Selection */}
          <Form.Item label="Mode" name="mode" className="mb-1">
            <AntdSelect
              style={{ width: "100%" }}
              value={testMode}
              onChange={(value) => setTestMode(value)}
              options={TEST_MODES}
            />
          </Form.Item>

          <Row>
            <Col span={10}></Col>
            <Col span={10}>
              <Text className="mb-5 mt-1">
                <strong>Optional</strong> - LiteLLM endpoint to use when health checking this model{" "}
                <Link href="https://docs.litellm.ai/docs/proxy/health#health" target="_blank">
                  Learn more
                </Link>
              </Text>
            </Col>
          </Row>

          {/* Credentials Section */}
          <div className="mb-4">
            <Typography.Text className="text-sm text-gray-500 mb-2">
              Either select existing credentials OR enter new provider credentials below
            </Typography.Text>
          </div>

          <Form.Item label="Existing Credentials" name="litellm_credential_name">
            <AntdSelect
              showSearch
              placeholder="Select or search for existing credentials"
              optionFilterProp="children"
              filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
              options={[
                { value: null, label: "None" },
                ...credentials.map((credential) => ({
                  value: credential.credential_name,
                  label: credential.credential_name,
                })),
              ]}
              allowClear
            />
          </Form.Item>

          <div className="flex items-center my-4">
            <div className="flex-grow border-t border-gray-200"></div>
            <span className="px-4 text-gray-500 text-sm">OR</span>
            <div className="flex-grow border-t border-gray-200"></div>
          </div>

          {/* Provider Specific Fields */}
          <Form.Item
            noStyle
            shouldUpdate={(prevValues, currentValues) =>
              prevValues.litellm_credential_name !== currentValues.litellm_credential_name
            }
          >
            {({ getFieldValue }) => {
              const credentialName = getFieldValue("litellm_credential_name")
              if (!credentialName) {
                return <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />
              }
              return (
                <div className="text-gray-500 text-sm text-center">
                  Using existing credentials - no additional provider fields needed
                </div>
              )
            }}
          </Form.Item>

          <div className="flex items-center my-4">
            <div className="flex-grow border-t border-gray-200"></div>
            <span className="px-4 text-gray-500 text-sm">Additional Model Info Settings</span>
            <div className="flex-grow border-t border-gray-200"></div>
          </div>

          {/* Team-only Model Switch */}
          <Form.Item
            label="Team-BYOK Model"
            tooltip="Only use this model + credential combination for this team."
            className="mb-4"
          >
            <Tooltip title={!premiumUser ? "Enterprise-only feature. Upgrade to premium." : ""} placement="top">
              <Switch
                checked={isTeamOnly}
                onChange={(checked) => {
                  setIsTeamOnly(checked)
                  if (!checked) {
                    form.setFieldValue("team_id", undefined)
                  }
                }}
                disabled={!premiumUser}
              />
            </Tooltip>
          </Form.Item>

          {/* Team Selection */}
          {isTeamOnly && (
            <Form.Item
              label="Select Team"
              name="team_id"
              className="mb-4"
              rules={[{ required: isTeamOnly && !isAdmin, message: "Please select a team." }]}
            >
              <TeamDropdown teams={teams} disabled={!premiumUser} />
            </Form.Item>
          )}

          {/* Model Access Groups */}
          {isAdmin && (
            <Form.Item
              label="Model Access Group"
              name="model_access_group"
              className="mb-4"
              tooltip="Use model access groups to give users access to select models."
            >
              <AntdSelect
                mode="tags"
                showSearch
                placeholder="Select existing groups or type to create new ones"
                optionFilterProp="children"
                tokenSeparators={[","]}
                options={modelAccessGroups.map((group) => ({
                  value: group,
                  label: group,
                }))}
                maxTagCount="responsive"
                allowClear
              />
            </Form.Item>
          )}

          {/* Advanced Settings */}
          <AdvancedSettings
            showAdvancedSettings={showAdvancedSettings}
            setShowAdvancedSettings={setShowAdvancedSettings}
            teams={teams}
          />

          {/* Action Buttons */}
          <div className="flex justify-between items-center mb-4">
            <Tooltip title="Get help on our github">
              <Typography.Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Typography.Link>
            </Tooltip>
            <div className="space-x-2">
              <Button onClick={handleTestConnection} loading={isTestingConnection}>
                Test Connect
              </Button>
              <Button type="primary" htmlType="submit">
                Add Model
              </Button>
            </div>
          </div>
        </Form>
      </Card>

      {/* Test Connection Modal */}
      <Modal
        title="Connection Test Results"
        open={isResultModalVisible}
        onCancel={() => {
          setIsResultModalVisible(false)
          setIsTestingConnection(false)
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              setIsResultModalVisible(false)
              setIsTestingConnection(false)
            }}
          >
            Close
          </Button>,
        ]}
        width={700}
      >
        {isResultModalVisible && (
          <ConnectionErrorDisplay
            key={connectionTestId}
            formValues={form.getFieldsValue()}
            accessToken={accessToken}
            testMode={testMode}
            modelName={form.getFieldValue("model_name") || form.getFieldValue("model")}
            onClose={() => {
              setIsResultModalVisible(false)
              setIsTestingConnection(false)
            }}
            onTestComplete={() => setIsTestingConnection(false)}
          />
        )}
      </Modal>
    </>
  )
}

export default AddModelForm
