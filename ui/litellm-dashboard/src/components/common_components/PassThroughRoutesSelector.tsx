import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import { getPassThroughEndpointsCall } from "../networking";
import i18n from "@/i18n/i18n";

interface PassThroughRoutesSelectorProps {
  onChange?: (selectedRoutes: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
  teamId?: string | null;
}

interface PassThroughEndpoint {
  path: string;
  methods?: string[];
}

const routeOption = (endpoint: PassThroughEndpoint): MultiSelectOption => ({
  label: endpoint.methods?.length ? `${endpoint.methods.join(", ")} ${endpoint.path}` : endpoint.path,
  value: endpoint.path,
});

const PassThroughRoutesSelector: React.FC<PassThroughRoutesSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = i18n.t("passThroughRoutesSelector.placeholder"),
  disabled = false,
  teamId,
}) => {
  const { t } = useTranslation();
  const [passThroughRoutes, setPassThroughRoutes] = useState<MultiSelectOption[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchPassThroughRoutes = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await getPassThroughEndpointsCall(accessToken, teamId);
        if (response.endpoints) {
          setPassThroughRoutes(response.endpoints.map(routeOption));
        }
      } catch (error) {
        console.error("Error fetching pass through routes:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchPassThroughRoutes();
  }, [accessToken, teamId]);

  return (
    <MultiSelect
      options={passThroughRoutes}
      value={value}
      onValueChange={(routes) => onChange?.(routes)}
      placeholder={placeholder}
      emptyText={t("passThroughRoutesSelector.empty")}
      loading={loading}
      allowCustomValues
      disabled={disabled}
      className={className}
    />
  );
};

export default PassThroughRoutesSelector;
