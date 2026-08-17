import { SearchSelect } from "@/components/shared/SearchSelect";
import React from "react";
import { ENDPOINT_OPTIONS } from "./chatConstants";
import { useTranslation } from "react-i18next";

interface EndpointSelectorProps {
  endpointType: string; // Accept string to avoid type conflicts
  onEndpointChange: (value: string) => void;
  className?: string;
}

const EndpointSelector: React.FC<EndpointSelectorProps> = ({ endpointType, onEndpointChange, className }) => {
  const { t } = useTranslation();

  return (
    <div className={className}>
      <SearchSelect
        value={endpointType}
        onValueChange={onEndpointChange}
        options={ENDPOINT_OPTIONS}
        placeholder={t("playground.configuration.endpointPlaceholder")}
      />
    </div>
  );
};

export default EndpointSelector;
