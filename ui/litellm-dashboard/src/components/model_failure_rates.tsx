import React from 'react';
import { Card, Title } from '@tremor/react';
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
        <span className="font-mono text-right" style={{ color: value >= 50 ? '#dc2626' : undefined }}>
          {value.toFixed(2)}%
        </span>
      );
    },
    meta: { align: 'right' },
  },
  {
    header: 'Failed',
    accessorKey: 'failedRequests',
    cell: (info) => Number(info.getValue()),
    meta: { align: 'right' },
  },
  {
    header: 'Total',
    accessorKey: 'totalRequests',
    cell: (info) => Number(info.getValue()),
    meta: { align: 'right' },
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
    <div>
      <DataTable
        data={failureRates}
        columns={columns}
        renderSubComponent={() => <></>}
        getRowCanExpand={() => false}
        isLoading={false}
        noDataMessage="No models with failures"
      />
    </div>
  );
}; 