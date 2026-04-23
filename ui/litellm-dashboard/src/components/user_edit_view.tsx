import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Form, Select } from "antd";
import { Info } from "lucide-react";
import React, { useState } from "react";
import { all_admin_roles } from "../utils/roles";
import BudgetDurationDropdown from "./common_components/budget_duration_dropdown";
import { getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import NumericalInput from "./shared/numerical_input";

interface UserEditViewProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  userData: any;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onSubmit: (values: any) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  teams: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  userModels: string[];
  possibleUIRoles: Record<string, Record<string, string>> | null;
  isBulkEdit?: boolean;
}

export function UserEditView({
  userData,
  onCancel,
  onSubmit,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  teams,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  accessToken,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  userID,
  userRole,
  userModels,
  possibleUIRoles,
  isBulkEdit = false,
}: UserEditViewProps) {
  const [form] = Form.useForm();
  const [unlimitedBudget, setUnlimitedBudget] = useState(false);

  React.useEffect(() => {
    const maxBudget = userData.user_info?.max_budget;
    const isUnlimited = maxBudget === null || maxBudget === undefined;
    setUnlimitedBudget(isUnlimited);

    form.setFieldsValue({
      user_id: userData.user_id,
      user_email: userData.user_info?.user_email,
      user_alias: userData.user_info?.user_alias,
      user_role: userData.user_info?.user_role,
      models: userData.user_info?.models || [],
      max_budget: isUnlimited ? "" : maxBudget,
      budget_duration: userData.user_info?.budget_duration,
      metadata: userData.user_info?.metadata
        ? JSON.stringify(userData.user_info.metadata, null, 2)
        : undefined,
    });
  }, [userData, form]);

  const handleUnlimitedBudgetChange = (checked: boolean) => {
    setUnlimitedBudget(checked);
    if (checked) {
      form.setFieldsValue({ max_budget: "" });
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleSubmit = (values: any) => {
    if (values.metadata && typeof values.metadata === "string") {
      try {
        values.metadata = JSON.parse(values.metadata);
      } catch (error) {
        console.error("Error parsing metadata JSON:", error);
        return;
      }
    }

    if (
      unlimitedBudget ||
      values.max_budget === "" ||
      values.max_budget === undefined
    ) {
      values.max_budget = null;
    }

    onSubmit(values);
  };

  return (
    <Form form={form} onFinish={handleSubmit} layout="vertical">
      {!isBulkEdit && (
        <Form.Item label="User ID" name="user_id">
          <Input disabled />
        </Form.Item>
      )}

      {!isBulkEdit && (
        <Form.Item label="Email" name="user_email">
          <Input />
        </Form.Item>
      )}

      <Form.Item label="User Alias" name="user_alias">
        <Input />
      </Form.Item>

      <Form.Item
        label={
          <span>
            Global Proxy Role{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  This is the role that the user will globally on the proxy.
                  This role is independent of any team/org specific roles.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
        name="user_role"
      >
        <Select>
          {possibleUIRoles &&
            Object.entries(possibleUIRoles).map(
              ([role, { ui_label, description }]) => (
                <Select.Option key={role} value={role} title={ui_label}>
                  <div className="flex">
                    {ui_label}{" "}
                    <p className="ml-2 text-muted-foreground text-xs">
                      {description}
                    </p>
                  </div>
                </Select.Option>
              ),
            )}
        </Select>
      </Form.Item>

      <Form.Item
        label={
          <span>
            Personal Models{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  Select which models this user can access outside of
                  team-scope. Choose &apos;All Proxy Models&apos; to grant
                  access to all models available on the proxy.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
        name="models"
      >
        <Select
          mode="multiple"
          placeholder="Select models"
          style={{ width: "100%" }}
          disabled={!all_admin_roles.includes(userRole || "")}
        >
          <Select.Option key="all-proxy-models" value="all-proxy-models">
            All Proxy Models
          </Select.Option>
          <Select.Option key="no-default-models" value="no-default-models">
            No Default Models
          </Select.Option>
          {userModels.map((model) => (
            <Select.Option key={model} value={model}>
              {getModelDisplayName(model)}
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      <Form.Item
        label={
          <div className="flex items-center gap-3">
            <span>Max Budget (USD)</span>
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={unlimitedBudget}
                onCheckedChange={(c) => handleUnlimitedBudgetChange(c === true)}
              />
              <span className="text-sm">Unlimited Budget</span>
            </label>
          </div>
        }
        name="max_budget"
        rules={[
          {
            validator: (_, value) => {
              if (
                !unlimitedBudget &&
                (value === "" || value === null || value === undefined)
              ) {
                return Promise.reject(
                  new Error(
                    "Please enter a budget or select Unlimited Budget",
                  ),
                );
              }
              return Promise.resolve();
            },
          },
        ]}
      >
        <NumericalInput
          step={0.01}
          precision={2}
          style={{ width: "100%" }}
          disabled={unlimitedBudget}
        />
      </Form.Item>

      <Form.Item label="Reset Budget" name="budget_duration">
        <BudgetDurationDropdown />
      </Form.Item>

      <Form.Item label="Metadata" name="metadata">
        <Textarea rows={4} placeholder="Enter metadata as JSON" />
      </Form.Item>

      <div className="flex justify-end space-x-2">
        <Button variant="secondary" type="button" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">Save Changes</Button>
      </div>
    </Form>
  );
}
