import React, { useCallback, useState, useEffect } from "react";
import { Form, Select } from "antd";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Policy } from "./types";
import {
  teamListCall,
  keyListCall,
  modelAvailableCall,
  estimateAttachmentImpactCall,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { buildAttachmentData } from "./build_attachment_data";
import ImpactPreviewAlert from "./impact_preview_alert";

interface AddAttachmentFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  policies: Policy[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [impactResult, setImpactResult] = useState<any>(null);
  const { userId, userRole } = useAuthorized();

  const loadTeamsKeysAndModels = useCallback(async () => {
    if (!accessToken) return;

    setIsLoadingTeams(true);
    try {
      const teamsResponse = await teamListCall(accessToken, null, userId);
      const teamsArray = Array.isArray(teamsResponse)
        ? teamsResponse
        : teamsResponse?.data || [];
      const teamAliases = teamsArray
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((t: any) => t.team_alias)
        .filter(Boolean);
      setAvailableTeams(teamAliases);
    } catch (error) {
      console.error("Failed to load teams:", error);
    } finally {
      setIsLoadingTeams(false);
    }

    setIsLoadingKeys(true);
    try {
      const keysResponse = await keyListCall(
        accessToken,
        null,
        null,
        null,
        null,
        null,
        1,
        100,
      );
      const keysArray = keysResponse?.keys || keysResponse?.data || [];
      const keyAliases = keysArray
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((k: any) => k.key_alias)
        .filter(Boolean);
      setAvailableKeys(keyAliases);
    } catch (error) {
      console.error("Failed to load keys:", error);
    } finally {
      setIsLoadingKeys(false);
    }

    setIsLoadingModels(true);
    try {
      const modelsResponse = await modelAvailableCall(
        accessToken,
        userId || "",
        userRole || "",
      );
      const modelsArray =
        modelsResponse?.data ||
        (Array.isArray(modelsResponse) ? modelsResponse : []);
      const modelIds = modelsArray
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((m: any) => m.id || m.model_name)
        .filter(Boolean);
      setAvailableModels(modelIds);
    } catch (error) {
      console.error("Failed to load models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  }, [accessToken, userId, userRole]);

  useEffect(() => {
    if (visible && accessToken) {
      loadTeamsKeysAndModels();
    }
  }, [visible, accessToken, loadTeamsKeysAndModels]);

  const resetForm = () => {
    form.resetFields();
    setScopeType("global");
    setImpactResult(null);
  };

  const handlePreviewImpact = async () => {
    if (!accessToken) return;
    try {
      await form.validateFields(["policy_names"]);
    } catch {
      return;
    }
    setIsEstimating(true);
    try {
      const { policy_names = [] } = form.getFieldsValue(true);
      const firstPolicy = policy_names?.[0];
      if (!firstPolicy) return;
      const data = buildAttachmentData(
        {
          ...form.getFieldsValue(true),
          policy_name: firstPolicy,
        },
        scopeType
      );
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

      const values = form.getFieldsValue(true);
      const selectedPolicyNames: string[] = values.policy_names || [];

      const results = await Promise.allSettled(
        selectedPolicyNames.map((policyName) => {
          const data = buildAttachmentData(
            {
              ...values,
              policy_name: policyName,
            },
            scopeType
          );
          return createAttachment(accessToken, data);
        })
      );

      const successCount = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.filter((r) => r.status === "rejected") as PromiseRejectedResult[];

      if (successCount > 0 && failed.length === 0) {
        NotificationsManager.success(
          successCount === 1
            ? "Attachment created successfully"
            : `${successCount} attachments created successfully`
        );
      } else if (successCount > 0 && failed.length > 0) {
        NotificationsManager.fromBackend(
          `${successCount} attachments created, ${failed.length} failed`
        );
      } else {
        throw new Error(
          failed[0]?.reason instanceof Error
            ? failed[0].reason.message
            : "Failed to create attachments"
        );
      }

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
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Create Policy Attachment</DialogTitle>
        </DialogHeader>
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            scope_type: "global",
          }}
        >
          <Form.Item
            name="policy_names"
            label="Policies"
            rules={[
              {
                required: true,
                message: "Please select at least one policy",
              },
            ]}
          >
            <Select
              mode="multiple"
              placeholder="Select policies to attach"
              options={policyOptions}
              showSearch
              filterOption={(input, option) =>
                (option?.label ?? "")
                  .toLowerCase()
                  .includes(input.toLowerCase())
              }
              style={{ width: "100%" }}
            />
          </Form.Item>

          <div className="flex items-center gap-2 mb-2">
            <span className="font-bold">Scope</span>
            <hr className="flex-1 border-border" />
          </div>

          <Form.Item label="Scope Type">
            <RadioGroup
              value={scopeType}
              onValueChange={(v) =>
                setScopeType(v as "global" | "specific")
              }
              className="flex flex-col gap-2"
            >
              <label className="flex items-center gap-2 cursor-pointer">
                <RadioGroupItem value="specific" />
                Specific (teams, keys, models, or tags)
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <RadioGroupItem value="global" />
                Global (applies to all requests)
              </label>
            </RadioGroup>
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
                <span className="text-xs text-muted-foreground">
                  Matches tags from key/team <code>metadata.tags</code> or
                  tags passed dynamically in the request body. Use{" "}
                  <code>*</code> as a suffix wildcard (e.g.,{" "}
                  <code>prod-*</code> matches <code>prod-us</code>,{" "}
                  <code>prod-eu</code>).
                </span>
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
            <Button variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            {scopeType === "specific" && (
              <Button
                variant="secondary"
                onClick={handlePreviewImpact}
                disabled={isEstimating}
              >
                {isEstimating ? "Estimating..." : "Estimate Impact"}
              </Button>
            )}
            <Button onClick={handleSubmit} disabled={isSubmitting}>
              {isSubmitting ? "Creating..." : "Create Attachment"}
            </Button>
          </div>
        </Form>
      </DialogContent>
    </Dialog>
  );
};

export default AddAttachmentForm;
