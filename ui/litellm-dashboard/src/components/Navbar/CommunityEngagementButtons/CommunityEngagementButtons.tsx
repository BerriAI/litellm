import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { GithubOutlined, SlackOutlined } from "@ant-design/icons";
import { Tooltip } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";

const iconBtnClass =
  "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700 cursor-pointer";

export const CommunityEngagementButtons: React.FC = () => {
  const { t } = useTranslation();
  const disableShowPrompts = useDisableShowPrompts();

  if (disableShowPrompts) {
    return null;
  }

  return (
    <div
      className="flex items-center gap-0.5 rounded-md border border-gray-200/80 bg-gray-50 px-0.5 py-0"
      aria-label={t("navbar.communityLinks")}
    >
      <Tooltip title={t("navbar.slackCommunity")}>
        <a
          href="https://www.litellm.ai/support"
          target="_blank"
          rel="noopener noreferrer"
          className={iconBtnClass}
          aria-label={t("navbar.joinSlack")}
        >
          <SlackOutlined className="text-lg" />
        </a>
      </Tooltip>
      <Tooltip title={t("navbar.githubRepo")}>
        <a
          href="https://github.com/BerriAI/litellm"
          target="_blank"
          rel="noopener noreferrer"
          className={iconBtnClass}
          aria-label={t("navbar.githubRepo")}
        >
          <GithubOutlined className="text-lg" />
        </a>
      </Tooltip>
    </div>
  );
};
