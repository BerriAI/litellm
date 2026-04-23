"use client";
import React, { useEffect, useState } from "react";
import {
  Modal,
  Drawer,
  Form,
  Input,
  InputNumber,
  Select,
  Popconfirm,
  message,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import { Button } from "@tremor/react";
import { PlusOutlined, DeleteOutlined, QuestionCircleOutlined, RobotOutlined } from "@ant-design/icons";
import { TrashIcon, SwitchVerticalIcon, ChevronUpIcon, ChevronDownIcon } from "@heroicons/react/outline";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { createLiteLLMEval, listLiteLLMEvals, deleteLiteLLMEval, updateLiteLLMEval, modelAvailableCall } from "../networking";

interface EvalCriterion {
  name: string;
  weight: number;
  description: string;
  threshold?: number;
}

interface LiteLLMEval {
  eval_id: string;
  eval_name: string;
  version: number;
  criteria: EvalCriterion[];
  judge_model: string;
  description?: string;
  overall_threshold?: number;
  max_iterations: number;
  created_at: string;
}

interface Props {
  accessToken: string | null;
  userRole?: string | null;
  availableModels?: string[];
}

export default function EvalsCatalog({ accessToken, userRole, availableModels: availableModelsProp = [] }: Props) {
  const [evals, setEvals] = useState<LiteLLMEval[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [editEval, setEditEval] = useState<LiteLLMEval | null>(null);
  const [editSaving, setEditSaving] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>(availableModelsProp);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [form] = Form.useForm();
  const [editForm] = Form.useForm();

  const fetchEvals = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const data = await listLiteLLMEvals(accessToken);
      setEvals(data);
    } catch (e: any) {
      message.error(`Failed to load evals: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    if (!accessToken || availableModelsProp.length > 0) return;
    try {
      const data = await modelAvailableCall(accessToken, "", "");
      const names: string[] = data?.data?.map((m: any) => m.id) ?? [];
      setAvailableModels(names);
    } catch {
      // best-effort
    }
  };

  useEffect(() => {
    fetchEvals();
    fetchModels();
  }, [accessToken]);

  const handleCreate = async (values: any) => {
    if (!accessToken) return;
    setSaving(true);
    try {
      await createLiteLLMEval(accessToken, {
        eval_name: values.eval_name,
        criteria: values.criteria || [],
        judge_model: values.judge_model,
        description: values.description,
        overall_threshold: values.overall_threshold ?? 80,
        max_iterations: values.max_iterations ?? 1,
      });
      message.success("Eval created");
      setModalOpen(false);
      form.resetFields();
      fetchEvals();
    } catch (e: any) {
      message.error(`Failed to create eval: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleEdit = (ev: LiteLLMEval) => {
    setEditEval(ev);
    editForm.setFieldsValue({
      eval_name: ev.eval_name,
      description: ev.description,
      judge_model: ev.judge_model,
      overall_threshold: ev.overall_threshold ?? 80,
      max_iterations: ev.max_iterations ?? 1,
      criteria: (ev.criteria || []).map((c: any) => ({
        name: c.name,
        weight: c.weight,
        description: c.description,
      })),
    });
  };

  const handleUpdate = async (values: any) => {
    if (!accessToken || !editEval) return;
    setEditSaving(true);
    try {
      await updateLiteLLMEval(accessToken, editEval.eval_id, {
        eval_name: values.eval_name,
        criteria: values.criteria || [],
        judge_model: values.judge_model,
        description: values.description,
        overall_threshold: values.overall_threshold ?? 80,
        max_iterations: values.max_iterations ?? 1,
      });
      message.success("Eval updated");
      setEditEval(null);
      editForm.resetFields();
      fetchEvals();
    } catch (e: any) {
      message.error(`Failed to update eval: ${e.message}`);
    } finally {
      setEditSaving(false);
    }
  };

  const handleDelete = async (evalId: string) => {
    if (!accessToken) return;
    try {
      await deleteLiteLLMEval(accessToken, evalId);
      message.success("Eval deleted");
      fetchEvals();
    } catch (e: any) {
      message.error(`Failed to delete eval: ${e.message}`);
    }
  };

  const columns: ColumnDef<LiteLLMEval>[] = [
    {
      header: "Name",
      accessorKey: "eval_name",
      cell: ({ row }) => (
        <span
          className="text-xs font-semibold text-blue-600 cursor-pointer hover:underline"
          onClick={() => handleEdit(row.original)}
        >
          {row.original.eval_name}
        </span>
      ),
    },
    {
      header: "Judge Model",
      accessorKey: "judge_model",
      cell: ({ row }) => (
        <span className="text-xs font-mono">{row.original.judge_model}</span>
      ),
    },
    {
      header: "Minimum Score to Pass",
      accessorKey: "overall_threshold",
      cell: ({ row }) => {
        const v = row.original.overall_threshold;
        return v != null ? (
          <Tag color="blue" className="text-xs">≥ {v} / 100</Tag>
        ) : (
          <span className="text-gray-400 text-xs">—</span>
        );
      },
    },
    {
      header: "Criteria",
      accessorKey: "criteria",
      cell: ({ row }) => (
        <Tag className="text-xs">{row.original.criteria?.length ?? 0} criteria</Tag>
      ),
    },
    {
      header: "Version",
      accessorKey: "version",
      cell: ({ row }) => (
        <span className="text-xs text-gray-500">v{row.original.version}</span>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => (
        <Popconfirm
          title="Delete this eval?"
          onConfirm={() => handleDelete(row.original.eval_id)}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <Button
            variant="light"
            size="xs"
            className="text-red-500 hover:text-red-700 hover:bg-red-50"
          >
            <TrashIcon className="h-4 w-4" />
          </Button>
        </Popconfirm>
      ),
    },
  ];

  const table = useReactTable({
    data: evals,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
  });

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-semibold">Evals</h1>
          <Tag color="blue" className="text-xs">Beta</Tag>
        </div>
        <Button
          className="flex-shrink-0"
          onClick={() => setModalOpen(true)}
        >
          + Create Eval
        </Button>
      </div>

      <div className="border rounded-lg overflow-hidden">
        <Table>
          <TableHead>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHeaderCell
                    key={header.id}
                    className={`py-1 ${header.column.getCanSort() ? "cursor-pointer select-none" : ""}`}
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center gap-1 text-xs font-medium">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getCanSort() && (
                        <span>
                          {header.column.getIsSorted() === "asc" ? (
                            <ChevronUpIcon className="h-4 w-4" />
                          ) : header.column.getIsSorted() === "desc" ? (
                            <ChevronDownIcon className="h-4 w-4" />
                          ) : (
                            <SwitchVerticalIcon className="h-4 w-4 text-gray-400" />
                          )}
                        </span>
                      )}
                    </div>
                  </TableHeaderCell>
                ))}
              </TableRow>
            ))}
          </TableHead>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="text-center py-4 text-gray-500 text-xs">
                  Loading...
                </TableCell>
              </TableRow>
            ) : evals.length > 0 ? (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} className="h-8">
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className={`py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap text-xs ${
                        cell.column.id === "actions"
                          ? "sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)]"
                          : ""
                      }`}
                    >
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="text-center py-8 text-gray-500 text-xs">
                  No evals yet. Create one to get started.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      <Modal
        title={
          <div className="flex items-center gap-2">
            <span>Create Eval</span>
            <Tag color="blue" className="text-xs">Beta</Tag>
          </div>
        }
        open={modalOpen}
        width={700}
        onCancel={() => {
          setModalOpen(false);
          form.resetFields();
        }}
        footer={
          <div className="flex justify-end gap-2">
            <Button
              variant="secondary"
              onClick={() => {
                setModalOpen(false);
                form.resetFields();
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={() => form.submit()}
              disabled={saving}
            >
              {saving ? "Creating..." : "Create Eval"}
            </Button>
          </div>
        }
        className="top-8"
        styles={{
          body: { padding: "24px" },
          header: { padding: "24px 24px 0 24px", border: "none" },
        }}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          {/* How it works banner */}
          <div
            style={{
              background: "#f0f5ff",
              border: "1px solid #adc6ff",
              borderRadius: 6,
              padding: "10px 14px",
              marginBottom: 20,
              display: "flex",
              gap: 10,
              alignItems: "flex-start",
            }}
          >
            <RobotOutlined style={{ color: "#2f54eb", marginTop: 2, flexShrink: 0 }} />
            <Typography.Text style={{ fontSize: 13, color: "#1d1d1d" }}>
              After each agent response, the <strong>Judge Model</strong> (an LLM you choose) reads the response and scores it 0–100 against each criterion. The weighted average of those scores is the <strong>final score</strong>. If it falls below the threshold, the response is blocked.
            </Typography.Text>
          </div>

          <Form.Item name="eval_name" label="Eval Name" rules={[{ required: true }]}>
            <Input placeholder="e.g. Insurance Claims QA" />
          </Form.Item>

          <Form.Item name="description" label="Description">
            <Input.TextArea rows={2} placeholder="What does this eval measure?" />
          </Form.Item>

          <Form.Item
            name="judge_model"
            label={
              <span>
                Judge Model&nbsp;
                <Tooltip title="The LLM that reads each agent response and grades it against your criteria. It never sees end-user data beyond what the agent returned.">
                  <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
                </Tooltip>
              </span>
            }
            rules={[{ required: true }]}
          >
            <Select
              showSearch
              placeholder="Select a model"
              options={availableModels.map((m) => ({ label: m, value: m }))}
            />
          </Form.Item>

          <Form.Item
            name="overall_threshold"
            label={
              <span>
                Minimum Score to Pass&nbsp;
                <Tooltip title="The judge scores the response 0–100. If the weighted average falls below this number the eval fails. 80 is a good starting point.">
                  <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
                </Tooltip>
              </span>
            }
            initialValue={80}
          >
            <InputNumber min={0} max={100} addonAfter="/ 100" style={{ width: "100%" }} />
          </Form.Item>

          <Form.Item
            label={
              <span>
                Evaluation Criteria&nbsp;
                <Tooltip title="Each criterion is a thing the judge checks. Give it a name, a weight (how much it counts toward the final score), and a description telling the judge what to look for. Weights should add up to 100%.">
                  <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
                </Tooltip>
              </span>
            }
          >
            <Form.List name="criteria">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <div
                      key={key}
                      style={{
                        border: "1px solid #f0f0f0",
                        borderRadius: 6,
                        padding: "12px 12px 0",
                        marginBottom: 8,
                      }}
                    >
                      <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
                        <Form.Item
                          {...restField}
                          name={[name, "name"]}
                          rules={[{ required: true, message: "Enter criterion name" }]}
                          style={{ flex: 2, marginBottom: 8 }}
                        >
                          <Input placeholder="Criterion name (e.g. Policy accuracy)" />
                        </Form.Item>
                        <Form.Item
                          {...restField}
                          name={[name, "weight"]}
                          label={
                            <Tooltip title="How much this criterion counts toward the final score. All weights should add up to 100%.">
                              <span style={{ fontSize: 12, color: "#595959" }}>
                                Weight <QuestionCircleOutlined style={{ color: "#bfbfbf" }} />
                              </span>
                            </Tooltip>
                          }
                          rules={[{ required: true, message: "Enter weight" }]}
                          style={{ flex: 1, marginBottom: 8 }}
                        >
                          <InputNumber
                            min={0}
                            max={100}
                            addonAfter="%"
                            style={{ width: "100%" }}
                            placeholder="e.g. 25"
                          />
                        </Form.Item>
                        <div style={{ marginBottom: 8 }}>
                          <Button
                            variant="light"
                            size="xs"
                            className="text-red-500 hover:text-red-700"
                            onClick={() => remove(name)}
                          >
                            ×
                          </Button>
                        </div>
                      </div>
                      <Form.Item
                        {...restField}
                        name={[name, "description"]}
                        rules={[{ required: true, message: "Describe what to check" }]}
                        style={{ marginBottom: 8 }}
                      >
                        <Input placeholder="What should the judge check for this criterion?" />
                      </Form.Item>
                    </div>
                  ))}
                  <Button
                    variant="secondary"
                    size="sm"
                    className="w-full mt-1"
                    onClick={() => add({ weight: 0 })}
                  >
                    <PlusOutlined className="mr-1" /> Add Criterion
                  </Button>
                  {fields.length > 0 && (
                    <Form.Item shouldUpdate noStyle>
                      {() => {
                        const allCriteria: any[] = form.getFieldValue("criteria") || [];
                        const weightTotal = allCriteria.reduce(
                          (sum: number, c: any) => sum + (Number(c?.weight) || 0),
                          0
                        );
                        const weightOk = weightTotal === 100;
                        return (
                          <div style={{ marginTop: 6, fontSize: 12, color: weightOk ? "#52c41a" : "#faad14" }}>
                            Weights total: {weightTotal}%{weightOk ? " ✓" : " — should add up to 100%"}
                          </div>
                        );
                      }}
                    </Form.Item>
                  )}
                </>
              )}
            </Form.List>
          </Form.Item>

          <Form.Item
            name="max_iterations"
            label={
              <span>
                Retry on fail (attempts)&nbsp;
                <Tooltip title="If the response fails the eval, the agent is called again with the judge's feedback injected. Set to 1 to disable retries.">
                  <QuestionCircleOutlined style={{ color: "#8c8c8c" }} />
                </Tooltip>
              </span>
            }
            initialValue={1}
          >
            <InputNumber min={1} max={5} style={{ width: "100%" }} />
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit Drawer */}
      <Drawer
        title={editEval ? `Edit: ${editEval.eval_name}` : "Edit Eval"}
        width={600}
        open={!!editEval}
        onClose={() => { setEditEval(null); editForm.resetFields(); }}
        footer={
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => { setEditEval(null); editForm.resetFields(); }}>
              Cancel
            </Button>
            <Button onClick={() => editForm.submit()} disabled={editSaving}>
              {editSaving ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        }
      >
        <Form form={editForm} layout="vertical" onFinish={handleUpdate}>
          <Form.Item name="eval_name" label="Eval Name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item name="judge_model" label="Judge Model" rules={[{ required: true }]}>
            <Select showSearch options={availableModels.map((m) => ({ label: m, value: m }))} />
          </Form.Item>
          <Form.Item name="overall_threshold" label="Minimum Score to Pass">
            <InputNumber min={0} max={100} addonAfter="/ 100" style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item label="Evaluation Criteria">
            <Form.List name="criteria">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <div key={key} style={{ border: "1px solid #f0f0f0", borderRadius: 6, padding: "12px 12px 0", marginBottom: 8 }}>
                      <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
                        <Form.Item {...restField} name={[name, "name"]} rules={[{ required: true, message: "Enter name" }]} style={{ flex: 2, marginBottom: 8 }}>
                          <Input placeholder="Criterion name" />
                        </Form.Item>
                        <Form.Item {...restField} name={[name, "weight"]} rules={[{ required: true, message: "Enter weight" }]} style={{ flex: 1, marginBottom: 8 }}>
                          <InputNumber min={0} max={100} addonAfter="%" style={{ width: "100%" }} />
                        </Form.Item>
                        <div style={{ marginBottom: 8 }}>
                          <Button variant="light" size="xs" className="text-red-500 hover:text-red-700" onClick={() => remove(name)}>×</Button>
                        </div>
                      </div>
                      <Form.Item {...restField} name={[name, "description"]} rules={[{ required: true, message: "Describe what to check" }]} style={{ marginBottom: 8 }}>
                        <Input placeholder="What should the judge check for this criterion?" />
                      </Form.Item>
                    </div>
                  ))}
                  <Button variant="secondary" size="sm" className="w-full mt-1" onClick={() => add({ weight: 0 })}>
                    <PlusOutlined className="mr-1" /> Add Criterion
                  </Button>
                  {fields.length > 0 && (
                    <Form.Item shouldUpdate noStyle>
                      {() => {
                        const c: any[] = editForm.getFieldValue("criteria") || [];
                        const total = c.reduce((s: number, x: any) => s + (Number(x?.weight) || 0), 0);
                        return (
                          <div style={{ marginTop: 6, fontSize: 12, color: total === 100 ? "#52c41a" : "#faad14" }}>
                            Weights total: {total}%{total === 100 ? " ✓" : " — should add up to 100%"}
                          </div>
                        );
                      }}
                    </Form.Item>
                  )}
                </>
              )}
            </Form.List>
          </Form.Item>
          <Form.Item name="max_iterations" label="Retry on fail (attempts)">
            <InputNumber min={1} max={5} style={{ width: "100%" }} />
          </Form.Item>
        </Form>
      </Drawer>
    </div>
  );
}
