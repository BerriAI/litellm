import React, { useState } from "react";
import { Modal, Form, Input, Button, Spin, List, Tag } from "antd";
import { CheckCircleFilled, CloseCircleFilled, ExclamationCircleFilled } from "@ant-design/icons";
import { registerClaudeCodeMarketplace, getClaudeCodePluginsList } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { MarketplaceSource, PluginListItem } from "@/components/claude_code_plugins/types";

interface AddMarketplaceFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

interface AddMarketplaceFormValues {
  source: string;
  name?: string;
}

type Step = "form" | "importing" | "result";

const AddMarketplaceForm: React.FC<AddMarketplaceFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const [form] = Form.useForm();
  const [step, setStep] = useState<Step>("form");
  const [marketplace, setMarketplace] = useState<MarketplaceSource | null>(null);
  const [loadedSkills, setLoadedSkills] = useState<PluginListItem[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const resetAndClose = () => {
    form.resetFields();
    setStep("form");
    setMarketplace(null);
    setLoadedSkills([]);
    setErrorMessage(null);
    onClose();
  };

  const handleCancel = () => {
    // Once import has started, the marketplace already exists server-side
    // regardless of whether this dialog stays open - closing early shouldn't
    // look like nothing happened, so still refresh the lists behind it.
    if (step !== "form") onSuccess();
    resetAndClose();
  };

  const handleSubmit = async (values: AddMarketplaceFormValues) => {
    if (!accessToken) {
      NotificationsManager.error("No access token available");
      return;
    }

    setStep("importing");
    setErrorMessage(null);
    try {
      const response = await registerClaudeCodeMarketplace(accessToken, {
        source: values.source.trim(),
        ...(values.name?.trim() ? { name: values.name.trim() } : {}),
      });
      setMarketplace(response.marketplace);

      const skillsResponse = await getClaudeCodePluginsList(accessToken, false);
      const prefix = `${response.marketplace.name}--`;
      setLoadedSkills(skillsResponse.plugins.filter((p: PluginListItem) => p.name.startsWith(prefix)));

      onSuccess();
      setStep("result");
    } catch (error) {
      console.error("Error registering marketplace:", error);
      setErrorMessage(error instanceof Error && error.message ? error.message : "Failed to import marketplace");
      setStep("result");
    }
  };

  const renderForm = () => (
    <Form form={form} layout="vertical" onFinish={handleSubmit} className="mt-4">
      <Form.Item
        label="Repository"
        name="source"
        rules={[{ required: true, message: "Please enter a repository (org/repo) or URL" }]}
        tooltip="A GitHub org/repo (e.g. anthropics/claude-code-marketplace) or a full git URL"
      >
        <Input placeholder="org/repo or https://github.com/org/repo" className="rounded-lg" />
      </Form.Item>

      <Form.Item
        label="Name (Optional)"
        name="name"
        tooltip="Marketplace identifier used to namespace its skills. Defaults to the repository name"
      >
        <Input placeholder="my-marketplace" className="rounded-lg" />
      </Form.Item>

      <Form.Item className="mb-0 mt-6">
        <div className="flex justify-end gap-2">
          <Button onClick={handleCancel}>Cancel</Button>
          <Button type="primary" htmlType="submit">
            Next
          </Button>
        </div>
      </Form.Item>
    </Form>
  );

  const renderImporting = () => (
    <div className="flex flex-col items-center justify-center gap-4 py-12">
      <Spin size="large" />
      <p className="text-gray-600">Importing marketplace and loading its skills…</p>
      <p className="text-xs text-gray-400">This can take a few seconds for repositories with many skills.</p>
    </div>
  );

  const renderResult = () => {
    if (errorMessage) {
      return (
        <div className="mt-4">
          <div className="flex items-start gap-2 rounded-lg bg-red-50 p-4">
            <CloseCircleFilled className="mt-0.5 text-red-500" />
            <div>
              <p className="font-medium text-red-700">Import failed</p>
              <p className="text-sm text-red-600">{errorMessage}</p>
            </div>
          </div>
          <div className="mt-6 flex justify-end gap-2">
            <Button onClick={resetAndClose}>Close</Button>
            <Button type="primary" onClick={() => setStep("form")}>
              Back
            </Button>
          </div>
        </div>
      );
    }

    const skippedCount = marketplace?.skipped_count ?? 0;

    return (
      <div className="mt-4">
        <div className="flex items-start gap-2 rounded-lg bg-green-50 p-4">
          <CheckCircleFilled className="mt-0.5 text-green-500" />
          <div>
            <p className="font-medium text-green-700">
              {loadedSkills.length} skill{loadedSkills.length === 1 ? "" : "s"} loaded from &quot;{marketplace?.name}
              &quot;
            </p>
            {skippedCount > 0 && (
              <p className="mt-1 flex items-center gap-1 text-sm text-amber-600">
                <ExclamationCircleFilled />
                {skippedCount} skill{skippedCount === 1 ? "" : "s"} could not be fetched after retries. Use &quot;Sync
                now&quot; on the marketplace to try again.
              </p>
            )}
          </div>
        </div>

        <div className="mt-4 max-h-80 overflow-y-auto rounded-lg border border-gray-200">
          <List
            size="small"
            dataSource={loadedSkills}
            renderItem={(skill) => (
              <List.Item className="px-4">
                <div className="flex w-full items-center justify-between gap-2">
                  <span className="font-mono text-sm">{skill.name}</span>
                  <Tag color={skill.enabled ? "green" : "default"}>{skill.enabled ? "public" : "private"}</Tag>
                </div>
              </List.Item>
            )}
          />
        </div>

        <div className="mt-6 flex justify-end">
          <Button type="primary" onClick={resetAndClose}>
            Done
          </Button>
        </div>
      </div>
    );
  };

  return (
    <Modal title="Add Marketplace" open={visible} onCancel={handleCancel} footer={null} width={560} className="top-8">
      {step === "form" && renderForm()}
      {step === "importing" && renderImporting()}
      {step === "result" && renderResult()}
    </Modal>
  );
};

export default AddMarketplaceForm;
