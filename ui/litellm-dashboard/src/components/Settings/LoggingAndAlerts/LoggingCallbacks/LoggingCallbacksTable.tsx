import { Button } from "@tremor/react";
import type { TableProps } from "antd";
import { Table } from "antd";
import Title from "antd/es/typography/Title";
import React from "react";
import { useTranslation } from "react-i18next";
import TableIconActionButton from "../../../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
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

const getCallbackModes = (t: ReturnType<typeof useTranslation>["t"]): { value: string; label: string }[] => [
  { value: "success", label: t("settingsPages.loggingCallbacksTable.modeSuccess") },
  { value: "failure", label: t("settingsPages.loggingCallbacksTable.modeFailure") },
  { value: "success_and_failure", label: t("settingsPages.loggingCallbacksTable.modeSuccessAndFailure") },
];

export const LoggingCallbacksTable: React.FC<LoggingCallbacksProps> = ({
  callbacks,
  availableCallbacks = {},
  onTest = () => {},
  onEdit = () => {},
  onDelete = () => {},
  onAdd = () => {},
}) => {
  const { t } = useTranslation();
  const callbackModes = React.useMemo(() => getCallbackModes(t), [t]);

  const columns: TableProps<CallbackRow>["columns"] = [
    {
      title: (
        <span className="font-medium text-gray-700">{t("settingsPages.loggingCallbacksTable.callbackNameColumn")}</span>
      ),
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
      title: <span className="font-medium text-gray-700">{t("settingsPages.loggingCallbacksTable.modeColumn")}</span>,
      key: "mode",
      render: (_: unknown, record: CallbackRow) => {
        const mode = record.mode || "success";
        const label = callbackModes.find((m) => m.value === mode)?.label || mode;
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
      title: <span className="font-medium text-gray-700 text-right w-full block">{t("common.actions")}</span>,
      key: "actions",
      align: "right",
      render: (_: unknown, record: CallbackRow) => (
        <div className="flex justify-end gap-2">
          <TableIconActionButton
            variant="Test"
            tooltipText={t("settingsPages.loggingCallbacksTable.testCallbackTooltip")}
            onClick={() => onTest(record)}
          />
          <TableIconActionButton
            variant="Edit"
            tooltipText={t("settingsPages.loggingCallbacksTable.editCallbackTooltip")}
            onClick={() => onEdit(record)}
          />
          <TableIconActionButton
            variant="Delete"
            tooltipText={t("settingsPages.loggingCallbacksTable.deleteCallbackTooltip")}
            onClick={() => onDelete(record)}
          />
        </div>
      ),
      width: 240,
    },
  ];
  return (
    <>
      <div className="w-full mt-4">
        <Button onClick={onAdd} className="mx-auto">
          {t("settingsPages.loggingCallbacksTable.addCallbackButton")}
        </Button>
        <div className="flex justify-between items-center my-2">
          <Title level={4}>{t("settingsPages.loggingCallbacksTable.activeCallbacksTitle")}</Title>
        </div>
        {callbacks.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-8 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="text-center">
              <h3 className="text-lg font-medium text-gray-700 mb-2">
                {t("settingsPages.loggingCallbacksTable.emptyStateTitle")}
              </h3>
              <p className="text-gray-500">{t("settingsPages.loggingCallbacksTable.emptyStateDesc")}</p>
            </div>
          </div>
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
