import React from 'react';
import { LogEntry } from './columns';
import { DataTable } from './table';
import { columns } from './columns';
import { Card, Title, Text, Metric, AreaChart } from '@tremor/react';

interface SessionViewProps {
  sessionId: string;
  logs: LogEntry[];
  onBack: () => void;
}

export const SessionView: React.FC<SessionViewProps> = ({ sessionId, logs, onBack }) => {
  // Calculate session metrics
  const totalCost = logs.reduce((sum, log) => sum + (log.spend || 0), 0);
  const totalTokens = logs.reduce((sum, log) => sum + (log.total_tokens || 0), 0);
  const startTime = logs.length > 0 ? new Date(logs[0].startTime) : new Date();
  const endTime = logs.length > 0 ? new Date(logs[logs.length - 1].endTime) : new Date();
  const durationMs = endTime.getTime() - startTime.getTime();
  const durationSec = (durationMs / 1000).toFixed(2);
  
  // Prepare data for the timeline chart
  const timelineData = logs.map(log => ({
    time: new Date(log.startTime).toISOString(),
    tokens: log.total_tokens || 0,
    cost: log.spend || 0,
  }));

  return (
    <div className="space-y-6">
      {/* Header with back button */}
      <div className="mb-8">
        <div className="flex items-center space-x-4">
          <button
            onClick={onBack}
            className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to All Logs
          </button>
        </div>
        <div className="mt-4">
          <h1 className="text-2xl font-semibold text-gray-900">Session Details</h1>
          <p className="text-sm text-gray-500 font-mono mt-1">{sessionId}</p>
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
          <Metric>${totalCost.toFixed(4)}</Metric>
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
              renderSubComponent={({ row }) => (
                <div className="p-4 bg-gray-50">
                  <pre className="whitespace-pre-wrap text-sm">
                    {JSON.stringify(row.original, null, 2)}
                  </pre>
                </div>
              )}
              getRowCanExpand={() => true}
            />
          </div>
    </div>
  );
};