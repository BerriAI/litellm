import React from "react";
import { useTranslation } from "react-i18next";
import { Button as TremorButton, Text } from "@tremor/react";
import { Input, Modal } from "antd";

interface PublishModalProps {
  visible: boolean;
  promptName: string;
  isSaving: boolean;
  onNameChange: (name: string) => void;
  onPublish: () => void;
  onCancel: () => void;
}

const PublishModal: React.FC<PublishModalProps> = ({
  visible,
  promptName,
  isSaving,
  onNameChange,
  onPublish,
  onCancel,
}) => {
  const { t } = useTranslation();
  return (
    <Modal
      title={t("promptsPage.publishModal.title")}
      open={visible}
      onCancel={onCancel}
      footer={[
        <div key="footer" className="flex justify-end gap-2">
          <TremorButton variant="secondary" onClick={onCancel}>
            {t("common.cancel")}
          </TremorButton>
          <TremorButton onClick={onPublish} loading={isSaving}>
            {t("promptsPage.publishModal.publish")}
          </TremorButton>
        </div>,
      ]}
    >
      <div className="py-4">
        <Text className="mb-2">{t("common.name")}</Text>
        <Input
          value={promptName}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder={t("promptsPage.publishModal.namePlaceholder")}
          onPressEnter={onPublish}
          autoFocus
        />
        <Text className="text-gray-500 text-xs mt-2">{t("promptsPage.publishModal.versionedHint")}</Text>
      </div>
    </Modal>
  );
};

export default PublishModal;
