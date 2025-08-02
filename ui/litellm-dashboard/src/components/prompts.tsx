import React, { useState, useEffect } from "react"
import { Card, Text } from "@tremor/react"
import { getPromptsList } from "./networking"
import PromptTable from "./prompts/prompt_table"

interface PromptsProps {
  accessToken: string | null
  userRole?: string
}

interface PromptItem {
  prompt_id?: string
  prompt_name: string | null
  prompt_info: Record<string, any>
  created_at?: string
  updated_at?: string
}

interface PromptsResponse {
  prompts: PromptItem[]
}

const PromptsPanel: React.FC<PromptsProps> = ({ accessToken, userRole }) => {
  const [promptsList, setPromptsList] = useState<PromptItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null)

  const fetchPrompts = async () => {
    if (!accessToken) {
      return
    }

    setIsLoading(true)
    try {
      const response: PromptsResponse = await getPromptsList(accessToken)
      console.log(`prompts: ${JSON.stringify(response)}`)
      setPromptsList(response.prompts)
    } catch (error) {
      console.error("Error fetching prompts:", error)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchPrompts()
  }, [accessToken])

  const handlePromptClick = (promptId: string) => {
    setSelectedPromptId(promptId)
    // For now, just log the click. In the future, this could open a detail view
    console.log(`Clicked prompt: ${promptId}`)
  }

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex justify-between items-center mb-4">
        <Text className="text-lg font-semibold">Prompts</Text>
      </div>

      <Card>
        <PromptTable
          promptsList={promptsList}
          isLoading={isLoading}
          onPromptClick={handlePromptClick}
        />
      </Card>
    </div>
  )
}

export default PromptsPanel