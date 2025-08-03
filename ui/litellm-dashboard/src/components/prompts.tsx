import React, { useState, useEffect } from "react"
import { Card, Text } from "@tremor/react"
import { getPromptsList, PromptSpec, ListPromptsResponse } from "./networking"
import PromptTable from "./prompts/prompt_table"
import PromptInfoView from "./prompts/prompt_info"
import { isAdminRole } from "@/utils/roles"

interface PromptsProps {
  accessToken: string | null
  userRole?: string
}

const PromptsPanel: React.FC<PromptsProps> = ({ accessToken, userRole }) => {
  const [promptsList, setPromptsList] = useState<PromptSpec[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedPromptId, setSelectedPromptId] = useState<string | null>(null)

  const isAdmin = userRole ? isAdminRole(userRole) : false

  const fetchPrompts = async () => {
    if (!accessToken) {
      return
    }

    setIsLoading(true)
    try {
      const response: ListPromptsResponse = await getPromptsList(accessToken)
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
  }

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      {selectedPromptId ? (
        <PromptInfoView
          promptId={selectedPromptId}
          onClose={() => setSelectedPromptId(null)}
          accessToken={accessToken}
          isAdmin={isAdmin}
        />
      ) : (
        <>
          <div className="flex justify-between items-center mb-4">
            <Text className="text-lg font-semibold">Prompts</Text>
          </div>

          <PromptTable
            promptsList={promptsList}
            isLoading={isLoading}
            onPromptClick={handlePromptClick}
          />
        </>
      )}
    </div>
  )
}

export default PromptsPanel