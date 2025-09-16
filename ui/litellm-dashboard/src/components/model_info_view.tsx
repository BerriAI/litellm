import React, { useState, useEffect } from "react"
import {
  Card,
  Title,
  Text,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Grid,
  Badge,
  Button as TremorButton,
  TextInput,
} from "@tremor/react"
import NumericalInput from "./shared/numerical_input"
import { ArrowLeftIcon, TrashIcon, KeyIcon } from "@heroicons/react/outline"
import {
  modelDeleteCall,
  modelUpdateCall,
  CredentialItem,
  credentialGetCall,
  credentialCreateCall,
  modelInfoCall,
  modelInfoV1Call,
  modelPatchUpdateCall,
  getGuardrailsList,
  arcagiGetConfig,
  arcagiRunBenchmark,
  arcagiStreamProgress,
  arcagiFetchReport,
} from "./networking"
import { Button, Form, Input, InputNumber, message, Select, Modal, Tooltip, Radio, Progress } from "antd"
import { InfoCircleOutlined } from "@ant-design/icons"
import ReactMarkdown from "react-markdown"
import EditModelModal from "./edit_model/edit_model_modal"
import { handleEditModelSubmit } from "./edit_model/edit_model_modal"
import { getProviderLogoAndName } from "./provider_info_helpers"
import { getDisplayModelName } from "./view_model/model_name_display"
import AddCredentialsModal from "./model_add/add_credentials_tab"
import ReuseCredentialsModal from "./model_add/reuse_credentials"
import CacheControlSettings from "./add_model/cache_control_settings"
import { CheckIcon, CopyIcon } from "lucide-react"
import { copyToClipboard as utilCopyToClipboard } from "../utils/dataUtils"
import EditAutoRouterModal from "./edit_auto_router/edit_auto_router_modal"
import NotificationsManager from "./molecules/notifications_manager"
import { getProxyBaseUrl } from "./networking"

interface ModelInfoViewProps {
  modelId: string
  onClose: () => void
  modelData: any
  accessToken: string | null
  userID: string | null
  userRole: string | null
  editModel: boolean
  setEditModalVisible: (visible: boolean) => void
  setSelectedModel: (model: any) => void
  onModelUpdate?: (updatedModel: any) => void
  modelAccessGroups: string[] | null
}

export default function ModelInfoView({
  modelId,
  onClose,
  modelData,
  accessToken,
  userID,
  userRole,
  editModel,
  setEditModalVisible,
  setSelectedModel,
  onModelUpdate,
  modelAccessGroups,
}: ModelInfoViewProps) {
  const [form] = Form.useForm()
  const [localModelData, setLocalModelData] = useState<any>(null)
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false)
  const [isCredentialModalOpen, setIsCredentialModalOpen] = useState(false)
  const [isDirty, setIsDirty] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [existingCredential, setExistingCredential] = useState<CredentialItem | null>(null)
  const [showCacheControl, setShowCacheControl] = useState(false)
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({})
  const [isAutoRouterModalOpen, setIsAutoRouterModalOpen] = useState(false)
  const [guardrailsList, setGuardrailsList] = useState<string[]>([])
  const canEditModel = userRole === "Admin" || modelData?.model_info?.created_by === userID
  const isAdmin = userRole === "Admin"
  const isAutoRouter = modelData?.litellm_params?.auto_router_config != null

  const usingExistingCredential =
    modelData?.litellm_params?.litellm_credential_name != null &&
    modelData?.litellm_params?.litellm_credential_name != undefined

  // ARC-AGI state
  const [arcagiDatasets, setArcagiDatasets] = useState<{ name: string; id: string; desc: string }[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("")
  const [isArcagiRunning, setIsArcagiRunning] = useState<boolean>(false)
  const [arcagiProgress, setArcagiProgress] = useState<number>(0)
  const [arcagiReportId, setArcagiReportId] = useState<string | null>(null)
  const [arcagiMarkdown, setArcagiMarkdown] = useState<string>("")
  const [arcagiError, setArcagiError] = useState<string>("")
  const [arcagiAbortController, setArcagiAbortController] = useState<AbortController | null>(null)

  useEffect(() => {
    const getExistingCredential = async () => {
      console.log("accessToken, ", accessToken)
      if (!accessToken) return
      if (usingExistingCredential) return
      let existingCredentialResponse = await credentialGetCall(accessToken, null, modelId)
      console.log("existingCredentialResponse, ", existingCredentialResponse)
      setExistingCredential({
        credential_name: existingCredentialResponse["credential_name"],
        credential_values: existingCredentialResponse["credential_values"],
        credential_info: existingCredentialResponse["credential_info"],
      })
    }

    const getModelInfo = async () => {
      if (!accessToken) return
      let modelInfoResponse = await modelInfoV1Call(accessToken, modelId)
      console.log("modelInfoResponse, ", modelInfoResponse)
      let specificModelData = modelInfoResponse.data[0]
      setLocalModelData(specificModelData)

      // Check if cache control is enabled
      if (specificModelData?.litellm_params?.cache_control_injection_points) {
        setShowCacheControl(true)
      }
    }

    const fetchGuardrails = async () => {
      if (!accessToken) return
      try {
        const response = await getGuardrailsList(accessToken)
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name)
        setGuardrailsList(guardrailNames)
      } catch (error) {
        console.error("Failed to fetch guardrails:", error)
      }
    }

    getExistingCredential()
    getModelInfo()
    fetchGuardrails()
  }, [accessToken, modelId])

  // Load ARC-AGI datasets
  useEffect(() => {
    const load = async () => {
      try {
        if (!accessToken) return
        const cfg = await arcagiGetConfig(accessToken)
        const ds = (cfg?.datasets || []) as any[]
        setArcagiDatasets(ds)
        if (ds.length > 0) setSelectedDatasetId(ds[0].id)
      } catch (e) {
        setArcagiError("Failed to load datasets")
      }
    }
    load()
    return () => {
      if (arcagiAbortController) arcagiAbortController.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken])

  const handleReuseCredential = async (values: any) => {
    console.log("values, ", values)
    if (!accessToken) return
    let credentialItem = {
      credential_name: values.credential_name,
      model_id: modelId,
      credential_info: {
        custom_llm_provider: localModelData.litellm_params?.custom_llm_provider,
      },
    }
    NotificationsManager.info("Storing credential..")
    let credentialResponse = await credentialCreateCall(accessToken, credentialItem)
    console.log("credentialResponse, ", credentialResponse)
    NotificationsManager.success("Credential stored successfully")
  }

  const handleModelUpdate = async (values: any) => {
    try {
      if (!accessToken) return
      setIsSaving(true)

      console.log("values.model_name, ", values.model_name)

      let updatedLitellmParams = {
        ...localModelData.litellm_params,
        model: values.litellm_model_name,
        api_base: values.api_base,
        custom_llm_provider: values.custom_llm_provider,
        organization: values.organization,
        tpm: values.tpm,
        rpm: values.rpm,
        max_retries: values.max_retries,
        timeout: values.timeout,
        stream_timeout: values.stream_timeout,
        input_cost_per_token: values.input_cost / 1_000_000,
        output_cost_per_token: values.output_cost / 1_000_000,
      }
      if (values.guardrails) {
        updatedLitellmParams.guardrails = values.guardrails
      }

      // Handle cache control settings
      if (values.cache_control && values.cache_control_injection_points?.length > 0) {
        updatedLitellmParams.cache_control_injection_points = values.cache_control_injection_points
      } else {
        delete updatedLitellmParams.cache_control_injection_points
      }

      // Parse the model_info from the form values
      let updatedModelInfo
      try {
        updatedModelInfo = values.model_info ? JSON.parse(values.model_info) : modelData.model_info
        // Update access_groups from the form
        if (values.model_access_group) {
          updatedModelInfo = {
            ...updatedModelInfo,
            access_groups: values.model_access_group,
          }
        }
      } catch (e) {
        NotificationsManager.fromBackend("Invalid JSON in Model Info")
        return
      }

      const updateData = {
        model_name: values.model_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      }

      await modelPatchUpdateCall(accessToken, updateData, modelId)

      const updatedModelData = {
        ...localModelData,
        model_name: values.model_name,
        litellm_model_name: values.litellm_model_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      }

      setLocalModelData(updatedModelData)

      if (onModelUpdate) {
        onModelUpdate(updatedModelData)
      }

      NotificationsManager.success("Model settings updated successfully")
      setIsDirty(false)
      setIsEditing(false)
    } catch (error) {
      console.error("Error updating model:", error)
      NotificationsManager.fromBackend("Failed to update model settings")
    } finally {
      setIsSaving(false)
    }
  }

  const handleRunArcagi = async () => {
    if (!accessToken || !modelData?.model_name) return
    if (!selectedDatasetId) {
      message.warning("Select a dataset first")
      return
    }
    setArcagiError("")
    setArcagiMarkdown("")
    setArcagiReportId(null)
    setArcagiProgress(0)
    setIsArcagiRunning(true)
    const ac = new AbortController()
    setArcagiAbortController(ac)
    try {
      const { job_id } = await arcagiRunBenchmark(accessToken, modelData.model_name, selectedDatasetId)
      await arcagiStreamProgress(
        accessToken,
        job_id,
        async (evt: any) => {
          if (evt?.type === "progress") {
            setArcagiProgress(Number(evt.percent || 0))
          } else if (evt?.type === "done") {
            const rid = evt.report_id as string
            setArcagiReportId(rid)
            try {
              const md = await arcagiFetchReport(accessToken, rid)
              setArcagiMarkdown(md)
            } catch (e) {
              setArcagiError("Failed to load final report")
            } finally {
              setIsArcagiRunning(false)
            }
          } else if (evt?.type === "error") {
            setArcagiError(evt.message || "Benchmark error")
            setIsArcagiRunning(false)
          }
        },
        ac.signal,
      )
    } catch (e: any) {
      setArcagiError(e?.message || "Failed to start benchmark")
      setIsArcagiRunning(false)
    }
  }

  const handleCopyReport = async () => {
    if (!arcagiMarkdown) return
    try {
      await navigator.clipboard.writeText(arcagiMarkdown)
      message.success("Report copied to clipboard")
    } catch {
      message.error("Failed to copy report")
    }
  }

  const handleSaveReport = async () => {
    if (!arcagiMarkdown) return
    const blob = new Blob([arcagiMarkdown], { type: "text/markdown" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `arcagi_report_${arcagiReportId || "latest"}.md`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const handleDelete = async () => {
    try {
      if (!accessToken) return
      await modelDeleteCall(accessToken, modelId)
      NotificationsManager.success("Model deleted successfully")

      if (onModelUpdate) {
        onModelUpdate({
          deleted: true,
          model_info: { id: modelId },
        })
      }

      onClose()
    } catch (error) {
      console.error("Error deleting the model:", error)
      NotificationsManager.fromBackend("Failed to delete model")
    }
  }

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text)
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }))
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }))
      }, 2000)
    }
  }

  const handleAutoRouterUpdate = (updatedModel: any) => {
    setLocalModelData(updatedModel)
    if (onModelUpdate) {
      onModelUpdate(updatedModel)
    }
  }

  if (!modelData) {
    return (
      <div className="p-4">
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Models
        </TremorButton>
        <Text>Model not found</Text>
      </div>
    )
  }

  const curlCommand = `curl -X POST "${getProxyBaseUrl()}/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $YOUR_API_KEY" \\
  -d '{
    "model": "${modelData.model_name}",
    "messages": [
      {
        "role": "user",
        "content": "Hello, describe your LLM params."
      }
    ]
  }'`

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
            Back to Models
          </TremorButton>
          <Title>Public Model Name: {getDisplayModelName(modelData)}</Title>
          <div className="flex items-center cursor-pointer">
            <Text className="text-gray-500 font-mono">{modelData.model_info.id}</Text>
            <Button
              type="text"
              size="small"
              icon={copiedStates["model-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(modelData.model_info.id, "model-id")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["model-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
        </div>
        <div className="flex gap-2">
          {isAdmin && (
            <TremorButton
              icon={KeyIcon}
              variant="secondary"
              onClick={() => setIsCredentialModalOpen(true)}
              className="flex items-center"
            >
              Re-use Credentials
            </TremorButton>
          )}
          {canEditModel && (
            <TremorButton
              icon={TrashIcon}
              variant="secondary"
              onClick={() => setIsDeleteModalOpen(true)}
              className="flex items-center"
            >
              Delete Model
            </TremorButton>
          )}
        </div>
      </div>

      <TabGroup>
        <TabList className="mb-6">
          <Tab>Overview</Tab>
          <Tab>Raw JSON</Tab>
          <Tab>ARC-AGI</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {/* Overview Grid */}
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6 mb-6">
              <Card>
                <Text>Provider</Text>
                <div className="mt-2 flex items-center space-x-2">
                  {modelData.provider && (
                    <img
                      src={getProviderLogoAndName(modelData.provider).logo}
                      alt={`${modelData.provider} logo`}
                      className="w-4 h-4"
                      onError={(e) => {
                        // Create a div with provider initial as fallback
                        const target = e.target as HTMLImageElement
                        const parent = target.parentElement
                        if (parent) {
                          const fallbackDiv = document.createElement("div")
                          fallbackDiv.className =
                            "w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs"
                          fallbackDiv.textContent = modelData.provider?.charAt(0) || "-"
                          parent.replaceChild(fallbackDiv, target)
                        }
                      }}
                    />
                  )}
                  <Title>{modelData.provider || "Not Set"}</Title>
                </div>
              </Card>
              <Card>
                <Text>LiteLLM Model</Text>
                <div className="mt-2 overflow-hidden">
                  <Tooltip title={modelData.litellm_model_name || "Not Set"}>
                    <div className="break-all text-sm font-medium leading-relaxed cursor-pointer">
                      {modelData.litellm_model_name || "Not Set"}
                    </div>
                  </Tooltip>
                </div>
              </Card>
              <Card>
                <Text>Pricing</Text>
                <div className="mt-2">
                  <Text>Input: ${modelData.input_cost}/1M tokens</Text>
                  <Text>Output: ${modelData.output_cost}/1M tokens</Text>
                </div>
              </Card>
            </Grid>

            <Card className="my-6">
              <div className="flex justify-between items-center py-1">
                <Text>Request Example:</Text>
                <Button
                  type="text"
                  size="small"
                  icon={copiedStates["curl-example"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
                  onClick={() => {
                    copyToClipboard(curlCommand, "curl-example")
                  }}
                  className={`transition-all duration-200 ${
                    copiedStates["curl-example"]
                      ? "text-green-400 bg-green-900/20 border-green-700"
                      : "text-blue-500 hover:text-blue-600 hover:bg-blue-900/20"
                  }`}
                />
              </div>

              <div className="relative pb-4">
                <pre className="bg-gray-900 p-4 rounded text-xs overflow-auto font-mono text-gray-300">
                  <code className="text-gray-300">{curlCommand}</code>
                </pre>
              </div>
            </Card>

            {/* Audit info shown as a subtle banner below the overview */}
            <div className="mb-6 text-sm text-gray-500 flex items-center gap-x-6">
              <div className="flex items-center gap-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Created At{" "}
                {modelData.model_info.created_at
                  ? new Date(modelData.model_info.created_at).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })
                  : "Not Set"}
              </div>
              <div className="flex items-center gap-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                  />
                </svg>
                Created By {modelData.model_info.created_by || "Not Set"}
              </div>
            </div>

            {/* Settings Card */}
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Model Settings</Title>
                <div className="flex gap-2">
                  {isAutoRouter && canEditModel && !isEditing && (
                    <TremorButton
                      variant="primary"
                      onClick={() => setIsAutoRouterModalOpen(true)}
                      className="flex items-center"
                    >
                      Edit Auto Router
                    </TremorButton>
                  )}
                  {canEditModel && !isEditing && (
                    <TremorButton variant="secondary" onClick={() => setIsEditing(true)} className="flex items-center">
                      Edit Model
                    </TremorButton>
                  )}
                </div>
              </div>
              {localModelData ? (
                <Form
                  form={form}
                  onFinish={handleModelUpdate}
                  initialValues={{
                    model_name: localModelData.model_name,
                    litellm_model_name: localModelData.litellm_model_name,
                    api_base: localModelData.litellm_params.api_base,
                    custom_llm_provider: localModelData.litellm_params.custom_llm_provider,
                    organization: localModelData.litellm_params.organization,
                    tpm: localModelData.litellm_params.tpm,
                    rpm: localModelData.litellm_params.rpm,
                    max_retries: localModelData.litellm_params.max_retries,
                    timeout: localModelData.litellm_params.timeout,
                    stream_timeout: localModelData.litellm_params.stream_timeout,
                    input_cost: localModelData.litellm_params.input_cost_per_token
                      ? localModelData.litellm_params.input_cost_per_token * 1_000_000
                      : localModelData.model_info?.input_cost_per_token * 1_000_000 || null,
                    output_cost: localModelData.litellm_params?.output_cost_per_token
                      ? localModelData.litellm_params.output_cost_per_token * 1_000_000
                      : localModelData.model_info?.output_cost_per_token * 1_000_000 || null,
                    cache_control: localModelData.litellm_params?.cache_control_injection_points ? true : false,
                    cache_control_injection_points: localModelData.litellm_params?.cache_control_injection_points || [],
                    model_access_group: Array.isArray(localModelData.model_info?.access_groups)
                      ? localModelData.model_info.access_groups
                      : [],
                    guardrails: Array.isArray(localModelData.litellm_params?.guardrails)
                      ? localModelData.litellm_params.guardrails
                      : [],
                  }}
                  layout="vertical"
                  onValuesChange={() => setIsDirty(true)}
                >
                  <div className="space-y-4">
                    <div className="space-y-4">
                      <div>
                        <Text className="font-medium">Model Name</Text>
                        {isEditing ? (
                          <Form.Item name="model_name" className="mb-0">
                            <TextInput placeholder="Enter model name" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">{localModelData.model_name}</div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">LiteLLM Model Name</Text>
                        {isEditing ? (
                          <Form.Item name="litellm_model_name" className="mb-0">
                            <TextInput placeholder="Enter LiteLLM model name" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">{localModelData.litellm_model_name}</div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Input Cost (per 1M tokens)</Text>
                        {isEditing ? (
                          <Form.Item name="input_cost" className="mb-0">
                            <NumericalInput placeholder="Enter input cost" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData?.litellm_params?.input_cost_per_token
                              ? (localModelData.litellm_params?.input_cost_per_token * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.input_cost_per_token
                                ? (localModelData.model_info.input_cost_per_token * 1_000_000).toFixed(4)
                                : null}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Output Cost (per 1M tokens)</Text>
                        {isEditing ? (
                          <Form.Item name="output_cost" className="mb-0">
                            <NumericalInput placeholder="Enter output cost" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData?.litellm_params?.output_cost_per_token
                              ? (localModelData.litellm_params.output_cost_per_token * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.output_cost_per_token
                                ? (localModelData.model_info.output_cost_per_token * 1_000_000).toFixed(4)
                                : null}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">API Base</Text>
                        {isEditing ? (
                          <Form.Item name="api_base" className="mb-0">
                            <TextInput placeholder="Enter API base" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.api_base || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Custom LLM Provider</Text>
                        {isEditing ? (
                          <Form.Item name="custom_llm_provider" className="mb-0">
                            <TextInput placeholder="Enter custom LLM provider" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.custom_llm_provider || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Organization</Text>
                        {isEditing ? (
                          <Form.Item name="organization" className="mb-0">
                            <TextInput placeholder="Enter organization" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.organization || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">TPM (Tokens per Minute)</Text>
                        {isEditing ? (
                          <Form.Item name="tpm" className="mb-0">
                            <NumericalInput placeholder="Enter TPM" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.tpm || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">RPM (Requests per Minute)</Text>
                        {isEditing ? (
                          <Form.Item name="rpm" className="mb-0">
                            <NumericalInput placeholder="Enter RPM" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.rpm || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Max Retries</Text>
                        {isEditing ? (
                          <Form.Item name="max_retries" className="mb-0">
                            <NumericalInput placeholder="Enter max retries" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.max_retries || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Timeout (seconds)</Text>
                        {isEditing ? (
                          <Form.Item name="timeout" className="mb-0">
                            <NumericalInput placeholder="Enter timeout" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.timeout || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Stream Timeout (seconds)</Text>
                        {isEditing ? (
                          <Form.Item name="stream_timeout" className="mb-0">
                            <NumericalInput placeholder="Enter stream timeout" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.stream_timeout || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Model Access Groups</Text>
                        {isEditing ? (
                          <Form.Item name="model_access_group" className="mb-0">
                            <Select
                              mode="tags"
                              showSearch
                              placeholder="Select existing groups or type to create new ones"
                              optionFilterProp="children"
                              tokenSeparators={[","]}
                              maxTagCount="responsive"
                              allowClear
                              style={{ width: "100%" }}
                              options={modelAccessGroups?.map((group) => ({
                                value: group,
                                label: group,
                              }))}
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.model_info?.access_groups ? (
                              Array.isArray(localModelData.model_info.access_groups) ? (
                                localModelData.model_info.access_groups.length > 0 ? (
                                  <div className="flex flex-wrap gap-1">
                                    {localModelData.model_info.access_groups.map((group: string, index: number) => (
                                      <span
                                        key={index}
                                        className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                                      >
                                        {group}
                                      </span>
                                    ))}
                                  </div>
                                ) : (
                                  "No groups assigned"
                                )
                              ) : (
                                localModelData.model_info.access_groups
                              )
                            ) : (
                              "Not Set"
                            )}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">
                          Guardrails{" "}
                          <Tooltip title="Apply safety guardrails to this model to filter content or enforce policies">
                            <a
                              href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </a>
                          </Tooltip>
                        </Text>
                        {isEditing ? (
                          <Form.Item name="guardrails" className="mb-0">
                            <Select
                              mode="tags"
                              showSearch
                              placeholder="Select existing guardrails or type to create new ones"
                              optionFilterProp="children"
                              tokenSeparators={[","]}
                              maxTagCount="responsive"
                              allowClear
                              style={{ width: "100%" }}
                              options={guardrailsList.map((name) => ({
                                value: name,
                                label: name,
                              }))}
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.guardrails ? (
                              Array.isArray(localModelData.litellm_params.guardrails) ? (
                                localModelData.litellm_params.guardrails.length > 0 ? (
                                  <div className="flex flex-wrap gap-1">
                                    {localModelData.litellm_params.guardrails.map(
                                      (guardrail: string, index: number) => (
                                        <span
                                          key={index}
                                          className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800"
                                        >
                                          {guardrail}
                                        </span>
                                      ),
                                    )}
                                  </div>
                                ) : (
                                  "No guardrails assigned"
                                )
                              ) : (
                                localModelData.litellm_params.guardrails
                              )
                            ) : (
                              "Not Set"
                            )}
                          </div>
                        )}
                      </div>

                      {/* Cache Control Section */}
                      {isEditing ? (
                        <CacheControlSettings
                          form={form}
                          showCacheControl={showCacheControl}
                          onCacheControlChange={(checked) => setShowCacheControl(checked)}
                        />
                      ) : (
                        <div>
                          <Text className="font-medium">Cache Control</Text>
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            {localModelData.litellm_params?.cache_control_injection_points ? (
                              <div>
                                <p>Enabled</p>
                                <div className="mt-2">
                                  {localModelData.litellm_params.cache_control_injection_points.map(
                                    (point: any, i: number) => (
                                      <div key={i} className="text-sm text-gray-600 mb-1">
                                        Location: {point.location},{point.role && <span> Role: {point.role}</span>}
                                        {point.index !== undefined && <span> Index: {point.index}</span>}
                                      </div>
                                    ),
                                  )}
                                </div>
                              </div>
                            ) : (
                              "Disabled"
                            )}
                          </div>
                        </div>
                      )}

                      <div>
                        <Text className="font-medium">Model Info</Text>
                        {isEditing ? (
                          <Form.Item name="model_info" className="mb-0">
                            <Input.TextArea
                              rows={4}
                              placeholder='{"gpt-4": 100, "claude-v1": 200}'
                              defaultValue={JSON.stringify(modelData.model_info, null, 2)}
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded">
                            <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">
                              {JSON.stringify(localModelData.model_info, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                      <div>
                        <Text className="font-medium">Team ID</Text>
                        <div className="mt-1 p-2 bg-gray-50 rounded">{modelData.model_info.team_id || "Not Set"}</div>
                      </div>
                    </div>

                    {isEditing && (
                      <div className="mt-6 flex justify-end gap-2">
                        <TremorButton
                          variant="secondary"
                          onClick={() => {
                            form.resetFields()
                            setIsDirty(false)
                            setIsEditing(false)
                          }}
                        >
                          Cancel
                        </TremorButton>
                        <TremorButton variant="primary" onClick={() => form.submit()} loading={isSaving}>
                          Save Changes
                        </TremorButton>
                      </div>
                    )}
                  </div>
                </Form>
              ) : (
                <Text>Loading...</Text>
              )}
            </Card>
          </TabPanel>

          <TabPanel>
            <Card>
              <pre className="bg-gray-100 p-4 rounded text-xs overflow-auto">{JSON.stringify(modelData, null, 2)}</pre>
            </Card>
          </TabPanel>

          <TabPanel>
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>ARC-AGI Benchmark</Title>
              </div>
              {arcagiError && <Text className="text-red-600 mb-2">{arcagiError}</Text>}
              {/* Model context */}
              <div className="mb-3 text-sm text-gray-600">
                Evaluating model:{" "}
                <span className="font-mono">
                  {localModelData?.litellm_model_name || modelData?.litellm_model_name || modelData?.model_name}
                </span>
              </div>
              <div className="mb-4">
                <Text className="mb-2">Select the dataset:</Text>
                <Radio.Group onChange={(e) => setSelectedDatasetId(e.target.value)} value={selectedDatasetId}>
                  {arcagiDatasets.map((d) => (
                    <div key={d.id} className="py-2">
                      <Radio value={d.id}>
                        <span className="font-medium">
                          {d.name} ({d.id})
                        </span>
                      </Radio>
                      <div className="pl-6 text-gray-500 text-sm leading-snug max-w-3xl">{d.desc || ""}</div>
                    </div>
                  ))}
                </Radio.Group>
              </div>
              <div className="flex items-center justify-between gap-4 mb-6">
                <div className="flex items-center gap-4">
                  <TremorButton onClick={handleRunArcagi} disabled={isArcagiRunning || !selectedDatasetId}>
                    {isArcagiRunning ? "Running..." : "Run Test"}
                  </TremorButton>
                  <div className="w-64">
                    <Progress
                      percent={arcagiProgress}
                      status={isArcagiRunning ? "active" : arcagiProgress === 100 ? "success" : "normal"}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <TremorButton variant="secondary" onClick={handleCopyReport} disabled={!arcagiMarkdown}>
                    Copy
                  </TremorButton>
                  <TremorButton variant="secondary" onClick={handleSaveReport} disabled={!arcagiMarkdown}>
                    Save
                  </TremorButton>
                </div>
              </div>
              {arcagiMarkdown && (
                <div className="border-t pt-4 markdown-wrap">
                  <ReactMarkdown
                    components={{
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-4 pl-3 text-gray-700 italic mb-2">{children}</blockquote>
                      ),
                      h1: ({ children }) => <h1 className="text-xl font-semibold mb-2">{children}</h1>,
                      h2: ({ children }) => <h2 className="text-lg font-semibold mt-4 mb-2">{children}</h2>,
                      h3: ({ children }) => <h3 className="font-semibold mt-4 mb-2">{children}</h3>,
                      h4: ({ children }) => <h4 className="font-semibold mt-3 mb-2">{children}</h4>,
                      p: ({ children }) => <p className="mb-2 leading-relaxed">{children}</p>,
                      ul: ({ children }) => <ul className="list-disc pl-6 mb-2">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal pl-6 mb-2">{children}</ol>,
                      li: ({ children }) => <li className="mb-2">{children}</li>,
                      hr: () => <hr className="my-6 border-gray-200" />,
                      pre: ({ children }) => <pre className="whitespace-pre-wrap break-words">{children}</pre>,
                      code: ({ children }) => <code className="px-0 py-0">{children}</code>,
                    }}
                  >
                    {arcagiMarkdown}
                  </ReactMarkdown>
                </div>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">
              &#8203;
            </span>

            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">Delete Model</h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">Are you sure you want to delete this model?</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button onClick={handleDelete} className="ml-2" danger>
                  Delete
                </Button>
                <Button onClick={() => setIsDeleteModalOpen(false)}>Cancel</Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {isCredentialModalOpen && !usingExistingCredential ? (
        <ReuseCredentialsModal
          isVisible={isCredentialModalOpen}
          onCancel={() => setIsCredentialModalOpen(false)}
          onAddCredential={handleReuseCredential}
          existingCredential={existingCredential}
          setIsCredentialModalOpen={setIsCredentialModalOpen}
        />
      ) : (
        <Modal
          open={isCredentialModalOpen}
          onCancel={() => setIsCredentialModalOpen(false)}
          title="Using Existing Credential"
        >
          <Text>{modelData.litellm_params.litellm_credential_name}</Text>
        </Modal>
      )}

      {/* Edit Auto Router Modal */}
      <EditAutoRouterModal
        isVisible={isAutoRouterModalOpen}
        onCancel={() => setIsAutoRouterModalOpen(false)}
        onSuccess={handleAutoRouterUpdate}
        modelData={localModelData || modelData}
        accessToken={accessToken || ""}
        userRole={userRole || ""}
      />
    </div>
  )
}
