import React, { useEffect, useState } from 'react';
import { Card } from "@tremor/react";
import {
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
  Text
} from "@tremor/react";
import { uiSpendLogsCall } from './networking';
import moment from 'moment';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getSortedRowModel,
  SortingState,
} from '@tanstack/react-table';

interface SpendLogsTableProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

interface LogEntry {
  request_id: string;
  api_key: string;
  model: string;
  api_base?: string;
  call_type: string;
  spend: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  startTime: string;
  endTime: string;
  user?: string;
  metadata?: Record<string, any>;
  cache_hit: string;
  cache_key?: string;
  request_tags?: Record<string, any>;
  requester_ip_address?: string;
  messages: string | any[] | Record<string, any>;
  response: string | any[] | Record<string, any>;
}

const formatMessage = (message: any): string => {
  if (!message) return 'N/A';
  if (typeof message === 'string') return message;
  if (typeof message === 'object') {
    // Handle the {text, type} object specifically
    if (message.text) return message.text;
    if (message.content) return message.content;
    return JSON.stringify(message);
  }
  return String(message);
};

const RequestViewer: React.FC<{ data: any }> = ({ data }) => {
  const formatData = (input: any) => {
    if (typeof input === 'string') {
      try {
        return JSON.parse(input);
      } catch {
        return input;
      }
    }
    return input;
  };

  return (
    <div className="p-6 bg-gray-50 space-y-6">
      {/* Request Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Request</h3>
          <div>
            <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">Expand</button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">JSON</button>
          </div>
        </div>
        <pre className="p-4 overflow-auto text-sm">
          {JSON.stringify(formatData(data.request), null, 2)}
        </pre>
      </div>

    
      {/* Response Card */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Response</h3>
          <div>
            <button className="mr-2 px-3 py-1 text-sm border rounded hover:bg-gray-50">Expand</button>
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">JSON</button>
          </div>
        </div>
        <pre className="p-4 overflow-auto text-sm">
          {JSON.stringify(formatData(data.response), null, 2)}
        </pre>
      </div>
      {/* Metadata Card */}
      {data.metadata && Object.keys(data.metadata).length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <div className="flex justify-between items-center p-4 border-b">
            <h3 className="text-lg font-medium">Metadata</h3>
            <div>
              <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">JSON</button>
            </div>
          </div>
          <pre className="p-4 overflow-auto text-sm">
            {JSON.stringify(data.metadata, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};

export const SpendLogsTable: React.FC<SpendLogsTableProps> = ({ accessToken, token, userRole, userID }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  const columnHelper = createColumnHelper<LogEntry>();

  const columns = [
    {
      header: 'Time',
      accessorKey: 'startTime',
      cell: (info: any) => (
        <span>{moment(info.getValue()).format('MMM DD HH:mm:ss')}</span>
      ),
    },
    {
      header: 'Request ID',
      accessorKey: 'request_id',
      cell: (info: any) => (
        <span className="font-mono text-xs">{String(info.getValue() || '')}</span>
      ),
    },
    {
      header: 'Type',
      accessorKey: 'call_type',
      cell: (info: any) => (
        <span>{String(info.getValue() || '')}</span>
      ),
    },
    {
      header: 'Request',
      accessorKey: 'messages',
      cell: (info: any) => {
        const messages = info.getValue();
        try {
          const content = typeof messages === 'string' ? JSON.parse(messages) : messages;
          let displayText = '';
          
          if (Array.isArray(content)) {
            displayText = formatMessage(content[0]?.content);
          } else {
            displayText = formatMessage(content);
          }
          
          return <span className="truncate max-w-md text-sm">{displayText}</span>;
        } catch (e) {
          return <span className="truncate max-w-md text-sm">{formatMessage(messages)}</span>;
        }
      },
    },
    {
      header: 'Model',
      accessorKey: 'model',
      cell: (info: any) => <span>{String(info.getValue() || '')}</span>,
    },
    {
      header: 'Tokens',
      accessorKey: 'total_tokens',
      cell: (info: any) => {
        const row = info.row.original;
        return (
          <span className="text-sm">
            {String(row.total_tokens || '0')}
            <span className="text-gray-400 text-xs ml-1">
              ({String(row.prompt_tokens || '0')}+{String(row.completion_tokens || '0')})
            </span>
          </span>
        );
      },
    },
    {
      header: 'User',
      accessorKey: 'user',
      cell: (info: any) => <span>{String(info.getValue() || '-')}</span>,
    },
    {
      header: 'Cost',
      accessorKey: 'spend',
      cell: (info: any) => <span>${(Number(info.getValue() || 0)).toFixed(6)}</span>,
    },
    {
      header: 'Tags',
      accessorKey: 'request_tags',
      cell: (info: any) => {
        const tags = info.getValue();
        if (!tags || Object.keys(tags).length === 0) return '-';
        return (
          <div className="flex flex-wrap gap-1">
            {Object.entries(tags).map(([key, value]) => (
              <span key={key} className="px-2 py-1 bg-gray-100 rounded-full text-xs">
                {key}: {String(value)}
              </span>
            ))}
          </div>
        );
      },
    },
  ];

  const table = useReactTable({
    data: logs,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  useEffect(() => {
    const fetchLogs = async () => {
      if (!accessToken || !token || !userRole || !userID) {
        console.log("got None values for one of accessToken, token, userRole, userID");
        return;
      }

      try {
        setLoading(true);
        // Get logs for last 24 hours using ISO string and proper date formatting
        const endTime = moment().format('YYYY-MM-DD HH:mm:ss');
        const startTime = moment().subtract(24, 'hours').format('YYYY-MM-DD HH:mm:ss');
        
        const data = await uiSpendLogsCall(
          accessToken,
          token,
          userRole,
          userID,
          startTime,
          endTime
        );

        console.log("response from uiSpendLogsCall:", data);

        // Transform the data and add unique keys
        const formattedLogs = data.map((log: LogEntry, index: number) => ({
          ...log,
          key: index.toString(),
        }));

        setLogs(formattedLogs);
      } catch (error) {
        console.error('Error fetching logs:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, [accessToken, token, userRole, userID]);

  if (!accessToken || !token || !userRole || !userID) {
    console.log("got None values for one of accessToken, token, userRole, userID");
    return null;
  }

  return (
    <div className="w-[90%] mx-auto px-4">
      <div className="bg-white rounded-lg shadow w-full">
        <div className="p-4 border-b flex justify-between items-center">
          <div className="flex space-x-4 items-center">
            <input
              type="text"
              placeholder="Search by request ID, model, or user..."
              className="px-4 py-2 border rounded-lg w-80"
            />
            <button className="px-4 py-2 border rounded-lg flex items-center gap-2">
              <span>Filters</span>
              <span className="text-xs bg-gray-100 px-2 py-1 rounded">0</span>
            </button>
            <select className="px-4 py-2 border rounded-lg">
              <option>Last 24 hours</option>
              <option>Last 7 days</option>
              <option>Last 30 days</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1 text-sm border rounded hover:bg-gray-50">Export</button>
          </div>
        </div>
        
        <div className="overflow-x-auto w-full">
          {logs.length > 0 ? (
            <Table className="w-full table-fixed">
              <TableHead>
                <TableRow>
                  {columns.map((column) => (
                    <TableHeaderCell 
                      key={column.header}
                      className={`${column.header === 'Request ID' ? 'w-32' : ''} whitespace-nowrap`}
                    >
                      {column.header}
                    </TableHeaderCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {logs.map((log) => (
                  <React.Fragment key={log.request_id || log.startTime}>
                    <TableRow 
                      className="cursor-pointer hover:bg-gray-50"
                      onClick={() => setExpandedRow(expandedRow === log.request_id ? null : log.request_id)}
                    >
                      {columns.map((column) => (
                        <TableCell 
                          key={column.header}
                          className={column.header === 'Request ID' ? 'w-32 truncate' : ''}
                        >
                          {column.cell ? 
                            column.cell({ 
                              getValue: () => log[column.accessorKey as keyof LogEntry],
                              row: { original: log }
                            }) : 
                            <span>{String(log[column.accessorKey as keyof LogEntry] ?? '')}</span>
                          }
                        </TableCell>
                      ))}
                    </TableRow>
                    {expandedRow === log.request_id && (
                      <TableRow>
                        <TableCell colSpan={columns.length} className="p-0">
                          <RequestViewer data={{
                            request: typeof log.messages === 'string' ? JSON.parse(log.messages) : log.messages,
                            response: typeof log.response === 'string' ? JSON.parse(log.response) : log.response,
                            metadata: log.metadata
                          }} />
                        </TableCell>
                      </TableRow>
                    )}
                  </React.Fragment>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="p-8 text-center text-gray-500">
              <p>No SpendLogs messages available.</p>
              <p className="text-sm mt-2">
                To enable this, set <code>`general_settings.store_prompts_in_spend_logs: true`</code> in your config.yaml
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpendLogsTable;