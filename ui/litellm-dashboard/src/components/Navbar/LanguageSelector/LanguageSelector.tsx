import { LANGUAGE_STORAGE_KEY, SUPPORTED_LANGUAGES } from "@/lib/i18n";
import { setLocalStorageItem } from "@/utils/localStorageUtils";
import { GlobalOutlined } from "@ant-design/icons";
import type { MenuProps } from "antd";
import { Button, Dropdown } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";

const LanguageSelector: React.FC = () => {
  const { t, i18n } = useTranslation();

  const items: MenuProps["items"] = SUPPORTED_LANGUAGES.map((lang) => ({
    key: lang.code,
    label: lang.label,
  }));

  const handleSelect: MenuProps["onClick"] = ({ key }) => {
    i18n.changeLanguage(key);
    // Only persist explicit choices, so browser-language auto-detection
    // keeps working for users who never picked a language
    setLocalStorageItem(LANGUAGE_STORAGE_KEY, key);
  };

  return (
    <Dropdown
      menu={{ items, selectedKeys: [i18n.language], onClick: handleSelect }}
      trigger={["click"]}
      placement="bottomRight"
    >
      <Button
        type="text"
        className="!flex h-9 w-9 items-center justify-center rounded-md text-gray-500 hover:bg-gray-100 hover:text-gray-700"
        aria-label={t("navbar.language")}
        aria-haspopup="menu"
      >
        <GlobalOutlined className="text-lg" />
      </Button>
    </Dropdown>
  );
};

export default LanguageSelector;
