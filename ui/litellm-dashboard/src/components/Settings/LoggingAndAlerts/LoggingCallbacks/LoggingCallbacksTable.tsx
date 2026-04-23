import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import React from "react";
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
  return (
    <div className="w-full mt-4">
      <Button onClick={onAdd} className="mx-auto">
        + Add Callback
      </Button>
      <div className="flex justify-between items-center my-2">
        <h3 className="text-lg font-semibold">Active Logging Callbacks</h3>
      </div>
      {callbacks.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-8 bg-muted border border-border rounded-lg">
          <div className="text-center">
            <h3 className="text-lg font-medium text-foreground mb-2">
              No callbacks configured
            </h3>
            <p className="text-muted-foreground">
              Add your first callback to start logging data to external
              services.
            </p>
          </div>
        </div>
      ) : (
        <div className="bg-background border border-border rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="font-medium">Callback Name</TableHead>
                <TableHead className="font-medium w-[240px]">Mode</TableHead>
                <TableHead className="font-medium text-right w-[240px]">
                  Actions
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {callbacks.map((cb) => {
                const record = cb as CallbackRow;
                const id = record.name;
                const displayName =
                  availableCallbacks[id]?.ui_callback_name || id;
                const mode = record.mode || "success";
                const label =
                  CALLBACK_MODES.find((m) => m.value === mode)?.label || mode;
                const badgeClass =
                  mode === "success"
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                    : mode === "failure"
                      ? "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300"
                      : "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300";
                return (
                  <TableRow key={record.name} className="hover:bg-muted">
                    <TableCell>
                      <div className="font-medium text-foreground">
                        {displayName}
                      </div>
                    </TableCell>
                    <TableCell>
                      <span
                        className={cn(
                          "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
                          badgeClass,
                        )}
                      >
                        {label}
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <TableIconActionButton
                          variant="Test"
                          tooltipText="Test Callback"
                          onClick={() => onTest(record)}
                        />
                        <TableIconActionButton
                          variant="Edit"
                          tooltipText="Edit Callback"
                          onClick={() => onEdit(record)}
                        />
                        <TableIconActionButton
                          variant="Delete"
                          tooltipText="Delete Callback"
                          onClick={() => onDelete(record)}
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
};
