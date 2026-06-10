import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Modal, Form, Select, Upload, Button, Divider } from "antd";
import { TextInput } from "@tremor/react";
import { UploadOutlined } from "@ant-design/icons";
import type { UploadFile, UploadProps } from "antd";
import { convertPromptFileToJson, createPromptCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { Option } = Select;

interface AddPromptFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

interface PromptFormData {
  prompt_id: string;
  prompt_integration: string;
  prompt_file?: File;
}

const AddPromptForm: React.FC<AddPromptFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [promptIntegration, setPromptIntegration] = useState<string>("dotprompt");

  const handleCancel = () => {
    form.resetFields();
    setFileList([]);
    setPromptIntegration("dotprompt");
    onClose();
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();

      console.log("values: ", values);
      if (!accessToken) {
        NotificationsManager.fromBackend(t("promptsPage.addPromptForm.accessTokenRequired"));
        return;
      }

      if (promptIntegration === "dotprompt" && fileList.length === 0) {
        NotificationsManager.fromBackend(t("promptsPage.addPromptForm.uploadPromptFileRequired"));
        return;
      }

      setLoading(true);

      let promptData: any = {};

      if (promptIntegration === "dotprompt" && fileList.length > 0) {
        // Convert the uploaded file to JSON
        const file = fileList[0].originFileObj as File;

        try {
          const conversionResult = await convertPromptFileToJson(accessToken, file);
          console.log("Conversion result:", conversionResult);

          // Prepare prompt data for creation
          promptData = {
            prompt_id: values.prompt_id,
            litellm_params: {
              prompt_integration: "dotprompt",
              prompt_id: conversionResult.prompt_id,
              prompt_data: conversionResult.json_data,
            },
            prompt_info: {
              prompt_type: "db",
            },
          };
        } catch (conversionError) {
          console.error("Error converting prompt file:", conversionError);
          NotificationsManager.fromBackend(t("promptsPage.addPromptForm.conversionFailed"));
          setLoading(false);
          return;
        }
      }

      // Create the prompt
      try {
        await createPromptCall(accessToken, promptData);
        NotificationsManager.success(t("promptsPage.addPromptForm.createSuccess"));
        handleCancel();
        onSuccess();
      } catch (createError) {
        console.error("Error creating prompt:", createError);
        NotificationsManager.fromBackend(t("promptsPage.addPromptForm.createFailed"));
      }
    } catch (error) {
      console.error("Form validation error:", error);
    } finally {
      setLoading(false);
    }
  };

  const uploadProps: UploadProps = {
    beforeUpload: (file) => {
      if (!file.name.endsWith(".prompt")) {
        NotificationsManager.fromBackend(t("promptsPage.addPromptForm.uploadPromptFileRequired"));
        return false;
      }
      return false;
    },
    fileList,
    onChange: ({ fileList: newFileList }) => {
      setFileList(newFileList.slice(-1)); // Keep only the last file
    },
    onRemove: () => {
      setFileList([]);
    },
  };

  return (
    <Modal
      title={t("promptsPage.addPromptForm.title")}
      open={visible}
      onCancel={handleCancel}
      footer={[
        <Button key="cancel" onClick={handleCancel}>
          {t("common.cancel")}
        </Button>,
        <Button key="submit" loading={loading} onClick={handleSubmit}>
          {t("promptsPage.addPromptForm.createPrompt")}
        </Button>,
      ]}
      width={600}
    >
      <Form form={form} layout="vertical" requiredMark={false}>
        <Form.Item
          label={t("promptsPage.addPromptForm.promptIdLabel")}
          name="prompt_id"
          rules={[
            { required: true, message: t("promptsPage.addPromptForm.promptIdRequired") },
            {
              pattern: /^[a-zA-Z0-9_-]+$/,
              message: t("promptsPage.addPromptForm.promptIdPattern"),
            },
          ]}
        >
          <TextInput placeholder={t("promptsPage.addPromptForm.promptIdPlaceholder")} />
        </Form.Item>

        <Form.Item
          label={t("promptsPage.addPromptForm.integrationLabel")}
          name="prompt_integration"
          initialValue="dotprompt"
        >
          <Select value={promptIntegration} onChange={setPromptIntegration}>
            <Option value="dotprompt">dotprompt</Option>
          </Select>
        </Form.Item>

        {promptIntegration === "dotprompt" && (
          <>
            <Divider />
            <Form.Item
              label={t("promptsPage.addPromptForm.fileLabel")}
              extra={t("promptsPage.addPromptForm.fileExtra")}
            >
              <Upload {...uploadProps}>
                <Button icon={<UploadOutlined />}>{t("promptsPage.addPromptForm.selectFile")}</Button>
              </Upload>
              {fileList.length > 0 && (
                <div className="mt-2 text-sm text-gray-600">
                  {t("promptsPage.addPromptForm.selected", { name: fileList[0].name })}
                </div>
              )}
            </Form.Item>
          </>
        )}
      </Form>
    </Modal>
  );
};

export default AddPromptForm;
