import React, { useState } from "react"
import { LogEntry } from "./columns"
import { DataTable } from "./table"
import { columns } from "./columns"
import { Card, Title, Text, Metric, AreaChart, Button as TremorButton } from "@tremor/react"
import { RequestViewer } from "./index"
import { formatNumberWithCommas } from "@/utils/dataUtils"
import { ArrowLeftIcon } from "@heroicons/react/outline"

interface SessionViewProps {
  sessionId: string
  logs: LogEntry[]
  onBack: () => void
}

export const SessionView: React.FC<SessionViewProps> = ({ sessionId, logs, onBack }) => {
  // Track which log row is expanded
  const [expandedRequestId, setExpandedRequestId] = useState<string | null>(null)
  // Calculate session metrics
  const totalCost = logs.reduce((sum, log) => sum + (log.spend || 0), 0)
  const totalTokens = logs.reduce((sum, log) => sum + (log.total_tokens || 0), 0)
  const startTime = logs.length > 0 ? new Date(logs[0].startTime) : new Date()
  const endTime = logs.length > 0 ? new Date(logs[logs.length - 1].endTime) : new Date()
  const durationMs = endTime.getTime() - startTime.getTime()
  const durationSec = (durationMs / 1000).toFixed(2)

  // Prepare data for the timeline chart
  const timelineData = logs.map((log) => ({
    time: new Date(log.startTime).toISOString(),
    tokens: log.total_tokens || 0,
    cost: log.spend || 0,
  }))

  return (
    <div className="space-y-6">
      {/* Header with back button */}
      <div className="mb-8">
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onBack} className="mb-4">
          Back to All Logs
        </TremorButton>
        <div className="mt-4">
          <h1 className="text-2xl font-semibold text-gray-900">Session Details</h1>
          <div className="space-y-2">
            <p className="text-sm text-gray-500 font-mono">{sessionId}</p>
            <a
              href="https://docs.litellm.ai/docs/proxy/ui_logs_sessions"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              Get started with session management here
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
            </a>
          </div>
        </div>
      </div>

      {/* Session Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <Text>Total Requests</Text>
          <Metric>{logs.length}</Metric>
        </Card>
        <Card>
          <Text>Total Cost</Text>
          <Metric>${formatNumberWithCommas(totalCost, 4)}</Metric>
        </Card>
        <Card>
          <Text>Total Tokens</Text>
          <Metric>{totalTokens}</Metric>
        </Card>
      </div>
      {/* Request Timeline */}
      <Title>Session Logs</Title>
      <div className="mt-4">
        <DataTable
          columns={columns}
          data={logs}
          renderSubComponent={RequestViewer}
          getRowCanExpand={() => true}
          loadingMessage="Loading logs..."
          noDataMessage="No logs found"
        />
      </div>
    </div>
  )
}
