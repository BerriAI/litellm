import { InfoCircleOutlined } from "@ant-design/icons";
import { Select as AntdSelect, Tooltip, Typography } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";

const { Text } = Typography;

export const DEFAULT_ESCALATION_KEYWORDS = ["LITELLM ESCALATE"];

interface EscalationKeywordsProps {
  keywords: string[];
  onChange: (keywords: string[]) => void;
}

const EscalationKeywords: React.FC<EscalationKeywordsProps> = ({ keywords, onChange }) => {
  const { t } = useTranslation("gateway");
  return (
    <div className="w-full max-w-none">
      <div className="flex items-center gap-2 mb-1">
        <Typography.Title level={4} style={{ margin: 0 }}>
          {t("models.autoRouters.details.escalation.title")}
        </Typography.Title>
        <Tooltip title={t("models.autoRouters.details.escalation.tooltip")}>
          <InfoCircleOutlined className="text-gray-400" />
        </Tooltip>
      </div>
      <Text type="secondary" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
        {t("models.autoRouters.details.escalation.description")}
      </Text>
      <AntdSelect
        mode="tags"
        value={keywords}
        onChange={onChange}
        placeholder={t("models.autoRouters.details.escalation.placeholder")}
        tokenSeparators={[","]}
        open={false}
        suffixIcon={null}
        style={{ width: "100%" }}
        allowClear
      />
    </div>
  );
};

export default EscalationKeywords;
