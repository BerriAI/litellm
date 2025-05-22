import React from 'react';
import { Card, Title, Text } from '@tremor/react';
import { ModelActivityData } from './usage/types';
import { DataTable } from './view_logs/table';
import type { ColumnDef } from '@tanstack/react-table';

interface ModelFailureRow {
  name: string;
  failureRate: number;
  failedRequests: number;
  totalRequests: number;
}

const columns: ColumnDef<ModelFailureRow>[] = [
  {
    header: 'Model',
    accessorKey: 'name',
    cell: (info) => (
      <span className="max-w-xs truncate block" title={String(info.getValue())}>{String(info.getValue())}</span>
    ),
  },
  {
    header: 'Failure Rate',
    accessorKey: 'failureRate',
    cell: (info) => {
      const value = Number(info.getValue());
      return (
        <div className="w-full flex items-center">
          <div className="mr-2">
            <div className="w-44 bg-gray-200 rounded-sm h-2">
              <div 
                className="h-2 rounded-sm" 
                style={{ 
                  width: `${Math.min(100, value)}%`,
                  backgroundColor: '#dc2626'
                }}
              ></div>
            </div>
          </div>
          <div className="w-16 text-right text-red-600 font-mono">
            {value.toFixed(1)}%
          </div>
        </div>
      );
    },
  },
  {
    header: 'Failed / Total',
    accessorKey: 'failedRequests',
    cell: (info) => {
      const row = info.row.original;
      return (
        <div className="ml-2 text-sm">
          {row.failedRequests} / {row.totalRequests}
        </div>
      );
    },
  },
];

export const ModelFailureRates = ({ modelMetrics }: { modelMetrics: Record<string, ModelActivityData> }) => {
  // Prepare data
  const failureRates: ModelFailureRow[] = Object.entries(modelMetrics)
    .filter(([modelName, metrics]) => metrics.total_requests > 0 && modelName.trim() !== '')
    .map(([modelName, metrics]) => {
      const totalRequests = metrics.total_requests;
      const failedRequests = metrics.total_failed_requests;
      const failureRate = totalRequests > 0 ? (failedRequests / totalRequests) * 100 : 0;
      return {
        name: metrics.label || modelName,
        failureRate,
        failedRequests,
        totalRequests,
      };
    })
    .sort((a, b) => b.failureRate - a.failureRate)
    .slice(0, 10);

  if (failureRates.length === 0) return null;

  return (
    <Card className="mt-6">
      <div className="mb-5">
        <Title className="text-lg">Top Models by Failure Rate</Title>
      </div>
      
      <div className="overflow-hidden border border-gray-200 rounded-md">
        <DataTable
          data={failureRates}
          columns={columns}
          renderSubComponent={() => <></>}
          getRowCanExpand={() => false}
          isLoading={false}
          noDataMessage="No models with failures"
        />
      </div>
    </Card>
  );
}; 