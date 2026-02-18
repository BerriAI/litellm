import React, { useState, useEffect } from "react";
import { Modal, Form, Select, Radio, Divider, Typography } from "antd";
import { Button } from "@tremor/react";
import { Policy } from "./types";
import { teamListCall, keyListCall, modelAvailableCall, estimateAttachmentImpactCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { buildAttachmentData } from "./build_attachment_data";
import ImpactPreviewAlert from "./impact_preview_alert";

const { Text } = Typography;

interface AddAttachmentFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  policies: Policy[];
  createAttachment: (accessToken: string, attachmentData: any) => Promise<any>;
}

const AddAttachmentForm: React.FC<AddAttachmentFormProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
  policies,
  createAttachment,
}) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [scopeType, setScopeType] = useState<"global" | "specific">("global");
  const [availableTeams, setAvailableTeams] = useState<string[]>([]);
  const [availableKeys, setAvailableKeys] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingTeams, setIsLoadingTeams] = useState(false);
  const [isLoadingKeys, setIsLoadingKeys] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isEstimating, setIsEstimating] = useState(false);
  const [impactResult, setImpactResult] = useState<any>(null);
  const { userId, userRole } = useAuthorized();

  useEffect(() => {
    if (visible && accessToken) {
      loadTeamsKeysAndModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, accessToken]);

  const loadTeamsKeysAndModels = async () => {
    if (!accessToken) return;

    // Load teams — teamListCall returns a plain array of team objects
    setIsLoadingTeams(true);
    try {
      const teamsResponse = await teamListCall(accessToken, null, userId);
      const teamsArray = Array.isArray(teamsResponse) ? teamsResponse : (teamsResponse?.data || []);
      const teamAliases = teamsArray
        .map((t: any) => t.team_alias)
        .filter(Boolean);
      setAvailableTeams(teamAliases);
    } catch (error) {
      console.error("Failed to load teams:", error);
    } finally {
      setIsLoadingTeams(false);
    }

    // Load keys — keyListCall returns {keys: [...], total_count, ...}
    setIsLoadingKeys(true);
    try {
      const keysResponse = await keyListCall(accessToken, null, null, null, null, null, 1, 100);
      const keysArray = keysResponse?.keys || keysResponse?.data || [];
      const keyAliases = keysArray
        .map((k: any) => k.key_alias)
        .filter(Boolean);
      setAvailableKeys(keyAliases);
    } catch (error) {
      console.error("Failed to load keys:", error);
    } finally {
      setIsLoadingKeys(false);
    }

    // Load models
    setIsLoadingModels(true);
    try {
      const modelsResponse = await modelAvailableCall(accessToken, userId || "", userRole || "");
      const modelsArray = modelsResponse?.data || (Array.isArray(modelsResponse) ? modelsResponse : []);
      const modelIds = modelsArray
        .map((m: any) => m.id || m.model_name)
        .filter(Boolean);
      setAvailableModels(modelIds);
    } catch (error) {
      console.error("Failed to load models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const resetForm = () => {
    form.resetFields();
    setScopeType("global");
    setImpactResult(null);
  };

  const getAttachmentData = () => buildAttachmentData(form.getFieldsValue(true), scopeType);

  const handlePreviewImpact = async () => {
    if (!accessToken) return;
    try {
      await form.validateFields(["policy_name"]);
    } catch {
      return;
    }
    setIsEstimating(true);
    try {
      const data = getAttachmentData();
      const result = await estimateAttachmentImpactCall(accessToken, data);
      setImpactResult(result);
    } catch (error) {
      console.error("Failed to estimate impact:", error);
    } finally {
      setIsEstimating(false);
    }
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true);
      await form.validateFields();

      if (!accessToken) {
        throw new Error("No access token available");
      }

      const data = getAttachmentData();
      await createAttachment(accessToken, data);
      NotificationsManager.success("Attachment created successfully");

      resetForm();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create attachment:", error);
      NotificationsManager.fromBackend(
        "Failed to create attachment: " + (error instanceof Error ? error.message : String(error))
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const policyOptions = policies.map((p) => ({
    label: p.policy_name,
    value: p.policy_name,
  }));

  return (
    <Modal
      title="Create Policy Attachment"
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={600}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          scope_type: "global",
        }}
      >
        <Form.Item
          name="policy_name"
          label="Policy"
          rules={[{ required: true, message: "Please select a policy" }]}
        >
          <Select
            placeholder="Select a policy to attach"
            options={policyOptions}
            showSearch
            filterOption={(input, option) =>
              (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
            }
            style={{ width: "100%" }}
          />
        </Form.Item>

        <Divider orientation="left">
          <Text strong>Scope</Text>
        </Divider>

        <Form.Item label="Scope Type">
          <Radio.Group
            value={scopeType}
            onChange={(e) => setScopeType(e.target.value)}
          >
            <Radio value="specific">Specific (teams, keys, models, or tags)</Radio>
            <Radio value="global">Global (applies to all requests)</Radio>
          </Radio.Group>
        </Form.Item>

        {scopeType === "specific" && (
          <>
            <Form.Item
              name="teams"
              label="Teams"
              tooltip="Select team aliases or enter custom patterns. Supports wildcards (e.g., healthcare-*)"
            >
              <Select
                mode="tags"
                placeholder={isLoadingTeams ? "Loading teams..." : "Select or enter team aliases"}
                loading={isLoadingTeams}
                options={availableTeams.map((team) => ({
                  label: team,
                  value: team,
                }))}
                tokenSeparators={[","]}
                showSearch
                filterOption={(input, option) =>
                  (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                }
                style={{ width: "100%" }}
              />
            </Form.Item>

            <Form.Item
              name="keys"
              label="Keys"
              tooltip="Select key aliases or enter custom patterns. Supports wildcards (e.g., dev-*)"
            >
              <Select
                mode="tags"
                placeholder={isLoadingKeys ? "Loading keys..." : "Select or enter key aliases"}
                loading={isLoadingKeys}
                options={availableKeys.map((key) => ({
                  label: key,
                  value: key,
                }))}
                tokenSeparators={[","]}
                showSearch
                filterOption={(input, option) =>
                  (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                }
                style={{ width: "100%" }}
              />
            </Form.Item>

            <Form.Item
              name="models"
              label="Models"
              tooltip="Model names this attachment applies to. Supports wildcards (e.g., gpt-4*). Leave empty to apply to all models."
            >
              <Select
                mode="tags"
                placeholder={isLoadingModels ? "Loading models..." : "Select or enter model names (e.g., gpt-4, bedrock/*)"}
                loading={isLoadingModels}
                options={availableModels.map((model) => ({
                  label: model,
                  value: model,
                }))}
                tokenSeparators={[","]}
                showSearch
                filterOption={(input, option) =>
                  (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                }
                style={{ width: "100%" }}
              />
            </Form.Item>

            <Form.Item
              name="tags"
              label="Tags"
              tooltip="Match against tags set in key or team metadata. Use exact values (e.g., healthcare) or wildcard patterns (e.g., health-*) where * matches any suffix."
              extra={
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Matches tags from key/team <code>metadata.tags</code> or tags passed dynamically in the request body. Use <code>*</code> as a suffix wildcard (e.g., <code>prod-*</code> matches <code>prod-us</code>, <code>prod-eu</code>).
                </Text>
              }
            >
              <Select
                mode="tags"
                placeholder="Type a tag and press Enter (e.g. healthcare, prod-*)"
                tokenSeparators={[",", " "]}
                notFoundContent={null}
                suffixIcon={null}
                open={false}
                style={{ width: "100%" }}
              />
            </Form.Item>
          </>
        )}

        {impactResult && <ImpactPreviewAlert impactResult={impactResult} />}

        <div className="flex justify-end space-x-2 mt-4">
          <Button variant="secondary" onClick={handleClose}>
            Cancel
          </Button>
          {scopeType === "specific" && (
            <Button variant="secondary" onClick={handlePreviewImpact} loading={isEstimating}>
              Estimate Impact
            </Button>
          )}
          <Button onClick={handleSubmit} loading={isSubmitting}>
            Create Attachment
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default AddAttachmentForm;
