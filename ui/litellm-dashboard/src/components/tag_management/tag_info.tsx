import React, { useState, useEffect } from "react";
import {
  Card,
  Text,
  Title,
  Button,
  Badge,
  Accordion,
  AccordionHeader,
  AccordionBody,
  Title as TremorTitle,
} from "@tremor/react";
import { Form, Input, Select as Select2, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Trans, useTranslation } from "react-i18next";
import { fetchUserModels } from "../organisms/create_key_button";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { tagInfoCall, tagUpdateCall } from "../networking";
import { Tag } from "./types";
import NotificationsManager from "../molecules/notifications_manager";
import NumericalInput from "../shared/numerical_input";
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CheckIcon, CopyIcon } from "lucide-react";
import { Button as AntdButton } from "antd";

interface TagInfoViewProps {
  tagId: string;
  onClose: () => void;
  accessToken: string | null;
  is_admin: boolean;
  editTag: boolean;
}

const TagInfoView: React.FC<TagInfoViewProps> = ({ tagId, onClose, accessToken, is_admin, editTag }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [tagDetails, setTagDetails] = useState<Tag | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(editTag);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const fetchTagDetails = async () => {
    if (!accessToken) return;
    try {
      const response = await tagInfoCall(accessToken, [tagId]);
      const tagData = response[tagId];
      if (tagData) {
        setTagDetails(tagData);
        if (editTag) {
          form.setFieldsValue({
            name: tagData.name,
            description: tagData.description,
            models: tagData.models,
            max_budget: tagData.litellm_budget_table?.max_budget,
            budget_duration: tagData.litellm_budget_table?.budget_duration,
          });
        }
      }
    } catch (error) {
      console.error("Error fetching tag details:", error);
      NotificationsManager.fromBackend(t("tagManagement.tagInfo.errorFetchingTagDetails", { error }));
    }
  };

  useEffect(() => {
    fetchTagDetails();
  }, [tagId, accessToken]);

  useEffect(() => {
    if (accessToken) {
      // Using dummy values for userID and userRole since they're required by the function
      // TODO: Pass these as props if needed for the actual API implementation
      fetchUserModels("dummy-user", "Admin", accessToken, setUserModels);
    }
  }, [accessToken]);

  const handleSave = async (values: any) => {
    if (!accessToken) return;
    try {
      await tagUpdateCall(accessToken, {
        name: values.name,
        description: values.description,
        models: values.models,
        max_budget: values.max_budget,
        tpm_limit: values.tpm_limit,
        rpm_limit: values.rpm_limit,
        budget_duration: values.budget_duration,
      });
      NotificationsManager.success(t("tagManagement.tagInfo.tagUpdatedSuccess"));
      setIsEditing(false);
      fetchTagDetails();
    } catch (error) {
      console.error("Error updating tag:", error);
      NotificationsManager.fromBackend(t("tagManagement.tagInfo.errorUpdatingTag", { error }));
    }
  };

  if (!tagDetails) {
    return <div>{t("common.loading")}</div>;
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">
            {t("tagManagement.tagInfo.backToTags")}
          </Button>
          <div className="flex items-center gap-2">
            <Text className="font-medium">{t("tagManagement.tagInfo.tagNameLabel")}</Text>
            <span className="font-mono px-2 py-1 bg-gray-100 rounded text-sm border border-gray-200">
              {tagDetails.name}
            </span>
            <AntdButton
              type="text"
              size="small"
              icon={copiedStates["tag-name"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(tagDetails.name, "tag-name")}
              className={`transition-all duration-200 ${
                copiedStates["tag-name"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
          <Text className="text-gray-500">{tagDetails.description || t("tagManagement.tagInfo.noDescription")}</Text>
        </div>
        {is_admin && !isEditing && (
          <Button onClick={() => setIsEditing(true)}>{t("tagManagement.tagInfo.editTagButton")}</Button>
        )}
      </div>

      {isEditing ? (
        <Card>
          <Form form={form} onFinish={handleSave} layout="vertical" initialValues={tagDetails}>
            <Form.Item
              label={t("tagManagement.tagInfo.tagNameLabel")}
              name="name"
              rules={[{ required: true, message: t("tagManagement.tagInfo.tagNameRequired") }]}
            >
              <Input className="rounded-md border-gray-300" />
            </Form.Item>

            <Form.Item label={t("common.description")} name="description">
              <Input.TextArea rows={4} />
            </Form.Item>

            <Form.Item
              label={
                <span>
                  {t("tagManagement.tagInfo.allowedModelsLabel")}
                  <Tooltip title={t("tagManagement.tagInfo.allowedModelsTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="models"
            >
              <Select2 mode="multiple" placeholder={t("tagManagement.tagInfo.selectModelsPlaceholder")}>
                {userModels.map((modelId) => (
                  <Select2.Option key={modelId} value={modelId}>
                    {getModelDisplayName(modelId)}
                  </Select2.Option>
                ))}
              </Select2>
            </Form.Item>

            <Accordion className="mt-4 mb-4">
              <AccordionHeader>
                <TremorTitle className="m-0">{t("tagManagement.tagInfo.budgetRateLimits")}</TremorTitle>
              </AccordionHeader>
              <AccordionBody>
                <Form.Item
                  label={
                    <span>
                      {t("tagManagement.tagInfo.maxBudgetLabel")}{" "}
                      <Tooltip title={t("tagManagement.tagInfo.maxBudgetTooltip")}>
                        <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                      </Tooltip>
                    </span>
                  }
                  name="max_budget"
                >
                  <NumericalInput step={0.01} precision={2} width={200} />
                </Form.Item>

                <Form.Item
                  label={
                    <span>
                      {t("tagManagement.tagInfo.resetBudgetLabel")}{" "}
                      <Tooltip title={t("tagManagement.tagInfo.resetBudgetTooltip")}>
                        <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                      </Tooltip>
                    </span>
                  }
                  name="budget_duration"
                >
                  <BudgetDurationDropdown onChange={(value) => form.setFieldValue("budget_duration", value)} />
                </Form.Item>

                <div className="mt-4 p-3 bg-gray-50 rounded-md border border-gray-200">
                  <p className="text-sm text-gray-600">
                    <Trans
                      i18nKey="tagManagement.tagInfo.tpmRpmNotSupported"
                      components={{
                        githubLink: (
                          <a
                            href="https://github.com/BerriAI/litellm/issues/new"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:text-blue-800 underline"
                          />
                        ),
                      }}
                    />
                  </p>
                </div>
              </AccordionBody>
            </Accordion>

            <div className="flex justify-end space-x-2">
              <Button onClick={() => setIsEditing(false)}>{t("common.cancel")}</Button>
              <Button type="submit">{t("tagManagement.tagInfo.saveChanges")}</Button>
            </div>
          </Form>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card>
            <Title>{t("tagManagement.tagInfo.tagDetails")}</Title>
            <div className="space-y-4 mt-4">
              <div>
                <Text className="font-medium">{t("common.name")}</Text>
                <Text>{tagDetails.name}</Text>
              </div>
              <div>
                <Text className="font-medium">{t("common.description")}</Text>
                <Text>{tagDetails.description || "-"}</Text>
              </div>
              <div>
                <Text className="font-medium">{t("tagManagement.tagInfo.allowedModelsLabel")}</Text>
                <div className="flex flex-wrap gap-2 mt-2">
                  {!tagDetails.models || tagDetails.models.length === 0 ? (
                    <Badge color="red">{t("tagManagement.tagInfo.allModels")}</Badge>
                  ) : (
                    tagDetails.models.map((modelId) => (
                      <Badge key={modelId} color="blue">
                        <Tooltip title={`ID: ${modelId}`}>{tagDetails.model_info?.[modelId] || modelId}</Tooltip>
                      </Badge>
                    ))
                  )}
                </div>
              </div>
              <div>
                <Text className="font-medium">{t("tagManagement.tagInfo.created")}</Text>
                <Text>{tagDetails.created_at ? new Date(tagDetails.created_at).toLocaleString() : "-"}</Text>
              </div>
              <div>
                <Text className="font-medium">{t("tagManagement.tagInfo.lastUpdated")}</Text>
                <Text>{tagDetails.updated_at ? new Date(tagDetails.updated_at).toLocaleString() : "-"}</Text>
              </div>
            </div>
          </Card>

          {tagDetails.litellm_budget_table && (
            <Card>
              <Title>{t("tagManagement.tagInfo.budgetRateLimits")}</Title>
              <div className="space-y-4 mt-4">
                {tagDetails.litellm_budget_table.max_budget !== undefined &&
                  tagDetails.litellm_budget_table.max_budget !== null && (
                    <div>
                      <Text className="font-medium">{t("tagManagement.tagInfo.maxBudget")}</Text>
                      <Text>${tagDetails.litellm_budget_table.max_budget}</Text>
                    </div>
                  )}
                {tagDetails.litellm_budget_table.budget_duration && (
                  <div>
                    <Text className="font-medium">{t("tagManagement.tagInfo.budgetDuration")}</Text>
                    <Text>{tagDetails.litellm_budget_table.budget_duration}</Text>
                  </div>
                )}
                {tagDetails.litellm_budget_table.tpm_limit !== undefined &&
                  tagDetails.litellm_budget_table.tpm_limit !== null && (
                    <div>
                      <Text className="font-medium">{t("tagManagement.tagInfo.tpmLimit")}</Text>
                      <Text>{tagDetails.litellm_budget_table.tpm_limit.toLocaleString()}</Text>
                    </div>
                  )}
                {tagDetails.litellm_budget_table.rpm_limit !== undefined &&
                  tagDetails.litellm_budget_table.rpm_limit !== null && (
                    <div>
                      <Text className="font-medium">{t("tagManagement.tagInfo.rpmLimit")}</Text>
                      <Text>{tagDetails.litellm_budget_table.rpm_limit.toLocaleString()}</Text>
                    </div>
                  )}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};

export default TagInfoView;
