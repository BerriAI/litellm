import React from "react";
import { Switch } from "antd";
import { useTranslation } from "react-i18next";

interface TagFilteringToggleProps {
  enabled: boolean;
  routerFieldsMetadata: { [key: string]: any };
  onToggle: (enabled: boolean) => void;
}

const TagFilteringToggle: React.FC<TagFilteringToggleProps> = ({ enabled, routerFieldsMetadata, onToggle }) => {
  const { t } = useTranslation();

  return (
    <div className="space-y-3 max-w-3xl">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <label className="text-xs font-medium text-gray-700 uppercase tracking-wide">
            {routerFieldsMetadata["enable_tag_filtering"]?.ui_field_name ||
              t("routerSettings.tagFilteringToggle.label")}
          </label>
          <p className="text-xs text-gray-500 mt-0.5">
            {routerFieldsMetadata["enable_tag_filtering"]?.field_description || ""}
            {routerFieldsMetadata["enable_tag_filtering"]?.link && (
              <>
                {" "}
                <a
                  href={routerFieldsMetadata["enable_tag_filtering"].link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 underline"
                >
                  {t("common.learnMore")}
                </a>
              </>
            )}
          </p>
        </div>
        <Switch checked={enabled} onChange={onToggle} className="ml-4" />
      </div>
    </div>
  );
};

export default TagFilteringToggle;
