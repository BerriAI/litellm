import { PencilAltIcon, PlayIcon, TrashIcon } from "@heroicons/react/outline";
import { Button, Icon } from "@tremor/react";
import type { TableProps } from "antd";
import { Table } from "antd";
import Title from "antd/es/typography/Title";
import { PlusCircle } from "lucide-react";
import React from "react";
import { AlertingObject } from "./types";

type LoggingCallbacksProps = {
  callbacks: AlertingObject[];
  availableCallbacks?: Record<
    string,
    {
      litellm_callback_name: string;
      litellm_callback_params: string[];
      ui_callback_name: string;
    }
  >;
  onTest?: (callback: AlertingObject) => void | Promise<void>;
  onEdit?: (callback: AlertingObject) => void;
  onDelete?: (callback: AlertingObject) => void;
  onAdd?: () => void;
};

type CallbackRow = AlertingObject & {
  id?: string;
  mode?: "success" | "failure" | "info" | string;
};

const CALLBACK_MODES: { value: string; label: string }[] = [
  { value: "success", label: "Success" },
  { value: "failure", label: "Failure" },
  { value: "success_and_failure", label: "Success & Failure" },
];

export const LoggingCallbacksTable: React.FC<LoggingCallbacksProps> = ({
  callbacks,
  availableCallbacks = {},
  onTest = () => {},
  onEdit = () => {},
  onDelete = () => {},
  onAdd = () => {},
}) => {
  const columns: TableProps<CallbackRow>["columns"] = [
    {
      title: <span className="font-medium text-gray-700">Callback Name</span>,
      dataIndex: "name",
      key: "name",
      render: (_: string, record: CallbackRow) => {
        const id = record.name;
        console.log("availableCallbacks", availableCallbacks);
        const displayName = availableCallbacks[id]?.ui_callback_name || id;
        return <div className="font-medium text-gray-800">{displayName}</div>;
      },
    },
    {
      title: <span className="font-medium text-gray-700">Mode</span>,
      key: "mode",
      render: (_: unknown, record: CallbackRow) => {
        const mode = record.mode || "success";
        const label = CALLBACK_MODES.find((m) => m.value === mode)?.label || mode;
        const badgeClass =
          mode === "success"
            ? "bg-green-100 text-green-800"
            : mode === "failure"
              ? "bg-red-100 text-red-800"
              : "bg-blue-100 text-blue-800";
        return (
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${badgeClass}`}>
            {label}
          </span>
        );
      },
      width: 240,
    },
    {
      title: <span className="font-medium text-gray-700 text-right w-full block">Actions</span>,
      key: "actions",
      align: "right",
      render: (_: unknown, record: CallbackRow) => (
        <div className="flex justify-end gap-2">
          <span>
            <Icon
              icon={PlayIcon}
              size="sm"
              className="cursor-pointer text-indigo-600 hover:text-indigo-700"
              onClick={() => onTest(record)}
            />
          </span>

          <span>
            <Icon
              icon={PencilAltIcon}
              size="sm"
              className="cursor-pointer text-indigo-600 hover:text-indigo-700"
              onClick={() => onEdit(record)}
            />
          </span>
          <span>
            <Icon
              icon={TrashIcon}
              size="sm"
              className="cursor-pointer text-indigo-600 hover:text-red-600"
              onClick={() => onDelete(record)}
            />
          </span>
        </div>
      ),
      width: 240,
    },
  ];
  return (
    <>
      <div className="w-full mt-4">
        <Button onClick={onAdd} className="mx-auto">
          + Add Callback
        </Button>
        <div className="flex justify-between items-center my-2">
          <Title level={4}>Active Logging Callbacks</Title>
        </div>
        {/* Empty state */}
        {callbacks.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-8 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="text-center">
              <h3 className="text-lg font-medium text-gray-700 mb-2">No callbacks configured</h3>
              <p className="text-gray-500 mb-4">Add your first callback to start logging data to external services.</p>
              <button
                onClick={onAdd}
                className="inline-flex items-center gap-2 px-4 py-2 bg-[#6366f1] text-white rounded-md hover:bg-[#5558eb] transition-colors"
              >
                <PlusCircle size={18} />
                Add Callback
              </button>
            </div>
          </div> /* Callbacks list */
        ) : (
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <Table
              columns={columns}
              dataSource={callbacks as CallbackRow[]}
              rowKey={(record) => record.name}
              pagination={false}
              rowClassName={() => "hover:bg-gray-50"}
            />
          </div>
        )}
      </div>
    </>
  );
};
