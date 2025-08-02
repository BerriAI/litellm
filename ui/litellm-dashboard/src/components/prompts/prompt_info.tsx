import React, { useState, useEffect } from "react"
import {
  Card,
  Title,
  Text,
  Grid,
  Badge,
  Button as TremorButton,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
} from "@tremor/react"
import { Button, message, Tooltip } from "antd"
import { ArrowLeftIcon } from "@heroicons/react/outline"
import { getPromptInfo, PromptSpec } from "@/components/networking"
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils"
import { CheckIcon, CopyIcon } from "lucide-react"

export interface PromptInfoProps {
  promptId: string
  onClose: () => void
  accessToken: string | null
  isAdmin: boolean
}

const PromptInfoView: React.FC<PromptInfoProps> = ({ promptId, onClose, accessToken, isAdmin }) => {
  const [promptData, setPromptData] = useState<PromptSpec | null>(null)
  const [rawApiResponse, setRawApiResponse] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({})

  const fetchPromptInfo = async () => {
    try {
      setLoading(true)
      if (!accessToken) return
      const response = await getPromptInfo(accessToken, promptId)
      setPromptData(response)
      setRawApiResponse(response) // Store the raw response for the Raw JSON tab
    } catch (error) {
      message.error("Failed to load prompt information")
      console.error("Error fetching prompt info:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPromptInfo()
  }, [promptId, accessToken])

  if (loading) {
    return <div className="p-4">Loading...</div>
  }

  if (!promptData) {
    return <div className="p-4">Prompt not found</div>
  }

  // Format date helper function
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-"
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text)
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }))
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }))
      }, 2000)
    }
  }

  return (
    <div className="p-4">
      <div>
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Prompts
        </TremorButton>
        <Title>Prompt Details</Title>
        <div className="flex items-center cursor-pointer">
          <Text className="text-gray-500 font-mono">{promptData.prompt_id}</Text>
          <Button
            type="text"
            size="small"
            icon={copiedStates["prompt-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
            onClick={() => copyToClipboard(promptData.prompt_id, "prompt-id")}
            className={`left-2 z-10 transition-all duration-200 ${
              copiedStates["prompt-id"]
                ? "text-green-600 bg-green-50 border-green-200"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
            }`}
          />
        </div>
      </div>

      <TabGroup>
        <TabList className="mb-4">
          <Tab key="overview">Overview</Tab>
          {isAdmin ? <Tab key="details">Details</Tab> : <></>}
          <Tab key="raw-json">Raw JSON</Tab>
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Prompt ID</Text>
                <div className="mt-2">
                  <Title className="font-mono text-sm">{promptData.prompt_id}</Title>
                </div>
              </Card>

              <Card>
                <Text>Prompt Type</Text>
                <div className="mt-2">
                  <Title>{promptData.prompt_info?.prompt_type || "-"}</Title>
                  <Badge color="blue" className="mt-1">
                    {promptData.prompt_info?.prompt_type || "Unknown"}
                  </Badge>
                </div>
              </Card>

              <Card>
                <Text>Created At</Text>
                <div className="mt-2">
                  <Title>{formatDate(promptData.created_at)}</Title>
                  <Text>Last Updated: {formatDate(promptData.updated_at)}</Text>
                </div>
              </Card>
            </Grid>

            {promptData.litellm_params && Object.keys(promptData.litellm_params).length > 0 && (
              <Card className="mt-6">
                <Text className="font-medium">LiteLLM Parameters</Text>
                <div className="mt-2 p-3 bg-gray-50 rounded-md">
                  <pre className="text-xs text-gray-800 whitespace-pre-wrap">
                    {JSON.stringify(promptData.litellm_params, null, 2)}
                  </pre>
                </div>
              </Card>
            )}
          </TabPanel>

          {/* Details Panel (only for admins) */}
          {isAdmin && (
            <TabPanel>
              <Card>
                <Title className="mb-4">Prompt Details</Title>
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Prompt ID</Text>
                    <div className="font-mono text-sm bg-gray-50 p-2 rounded">{promptData.prompt_id}</div>
                  </div>
                  
                  <div>
                    <Text className="font-medium">Prompt Type</Text>
                    <div>{promptData.prompt_info?.prompt_type || "-"}</div>
                  </div>

                  <div>
                    <Text className="font-medium">Created At</Text>
                    <div>{formatDate(promptData.created_at)}</div>
                  </div>
                  
                  <div>
                    <Text className="font-medium">Last Updated</Text>
                    <div>{formatDate(promptData.updated_at)}</div>
                  </div>

                  <div>
                    <Text className="font-medium">LiteLLM Parameters</Text>
                    <div className="mt-2 p-3 bg-gray-50 rounded-md border">
                      <pre className="text-xs text-gray-800 whitespace-pre-wrap overflow-auto max-h-96">
                        {JSON.stringify(promptData.litellm_params, null, 2)}
                      </pre>
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Prompt Info</Text>
                    <div className="mt-2 p-3 bg-gray-50 rounded-md border">
                      <pre className="text-xs text-gray-800 whitespace-pre-wrap">
                        {JSON.stringify(promptData.prompt_info, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>
              </Card>
            </TabPanel>
          )}

          {/* Raw JSON Panel */}
          <TabPanel>
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Raw API Response</Title>
                <Button
                  type="text"
                  size="small"
                  icon={copiedStates["raw-json"] ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                  onClick={() => copyToClipboard(JSON.stringify(rawApiResponse, null, 2), "raw-json")}
                  className={`transition-all duration-200 ${
                    copiedStates["raw-json"]
                      ? "text-green-600 bg-green-50 border-green-200"
                      : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  {copiedStates["raw-json"] ? "Copied!" : "Copy JSON"}
                </Button>
              </div>
              
              <div className="p-4 bg-gray-50 rounded-md border overflow-auto">
                <pre className="text-xs text-gray-800 whitespace-pre-wrap">
                  {JSON.stringify(rawApiResponse, null, 2)}
                </pre>
              </div>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  )
}

export default PromptInfoView