/**
 * Unified selector component that handles both model and agent selection
 * based on the current endpoint configuration.
 */

import { Select, Spin } from "antd";
import { SelectorOption, EndpointConfig } from "../endpoint_config";
import { useTranslation } from "react-i18next";

interface UnifiedSelectorProps {
  value: string;
  options: SelectorOption[];
  loading: boolean;
  config: EndpointConfig;
  onChange: (value: string) => void;
}

export function UnifiedSelector({ value, options, loading, config, onChange }: UnifiedSelectorProps) {
  const { t } = useTranslation("chat");
  return (
    <Select
      value={value || undefined}
      placeholder={
        loading
          ? t("playground.compare.loadingSelector", { item: config.selectorLabel.toLowerCase() })
          : config.selectorPlaceholder
      }
      onChange={onChange}
      loading={loading}
      showSearch
      filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
      options={options}
      className="w-48 md:w-64 lg:w-72"
      notFoundContent={
        loading ? (
          <div className="flex items-center justify-center py-2">
            <Spin size="small" />
          </div>
        ) : (
          t("playground.compare.noSelectorOptions", { item: config.selectorLabel.toLowerCase() })
        )
      }
    />
  );
}
