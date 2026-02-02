import React, { useState, useEffect } from "react";
import { Modal, Form, Select, Radio, Divider, Typography } from "antd";
import { Button } from "@tremor/react";
import { Policy, PolicyAttachmentCreateRequest } from "./types";
import { teamListCall, keyInfoCall, modelAvailableCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

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
  const { userId, userRole } = useAuthorized();

  useEffect(() => {
    if (visible && accessToken) {
      loadTeamsKeysAndModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, accessToken]);

  const loadTeamsKeysAndModels = async () => {
    if (!accessToken) return;

    // Load teams
    setIsLoadingTeams(true);
    try {
      // Pass null for organizationID since we're loading all teams the user has access to
      const teamsResponse = await teamListCall(accessToken, null, userId);
      if (teamsResponse?.data) {
        const teamAliases = teamsResponse.data
          .map((t: any) => t.team_alias)
          .filter(Boolean);
        setAvailableTeams(teamAliases);
      }
    } catch (error) {
      console.error("Failed to load teams:", error);
    } finally {
      setIsLoadingTeams(false);
    }

    // Load keys
    setIsLoadingKeys(true);
    try {
      const keysResponse = await keyInfoCall(accessToken, []);
      if (keysResponse?.data) {
        const keyAliases = keysResponse.data
          .map((k: any) => k.key_alias)
          .filter(Boolean);
        setAvailableKeys(keyAliases);
      }
    } catch (error) {
      console.error("Failed to load keys:", error);
    } finally {
      setIsLoadingKeys(false);
    }

    // Load models
    setIsLoadingModels(true);
    try {
      const modelsResponse = await modelAvailableCall(accessToken, userId || "", userRole || "");
      if (modelsResponse?.data) {
        const modelIds = modelsResponse.data
          .map((m: any) => m.id || m.model_name)
          .filter(Boolean);
        setAvailableModels(modelIds);
      }
    } catch (error) {
      console.error("Failed to load models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const resetForm = () => {
    form.resetFields();
    setScopeType("global");
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true);
      await form.validateFields();
      const values = form.getFieldsValue(true);

      if (!accessToken) {
        throw new Error("No access token available");
      }

      const data: PolicyAttachmentCreateRequest = {
        policy_name: values.policy_name,
      };

      if (scopeType === "global") {
        data.scope = "*";
      } else {
        if (values.teams && values.teams.length > 0) {
          data.teams = values.teams;
        }
        if (values.keys && values.keys.length > 0) {
          data.keys = values.keys;
        }
        if (values.models && values.models.length > 0) {
          data.models = values.models;
        }
      }

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
            <Radio value="global">Global (applies to all requests)</Radio>
            <Radio value="specific">Specific (teams, keys, or models)</Radio>
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
          </>
        )}

        <div className="flex justify-end space-x-2 mt-4">
          <Button variant="secondary" onClick={handleClose}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} loading={isSubmitting}>
            Create Attachment
          </Button>
        </div>
      </Form>
    </Modal>
  );
};

export default AddAttachmentForm;
