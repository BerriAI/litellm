import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { getPassThroughEndpointsCall } from "../networking";

interface PassThroughRoutesSelectorProps {
  onChange: (selectedRoutes: string[]) => void;
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

const PassThroughRoutesSelector: React.FC<PassThroughRoutesSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select pass through routes",
  disabled = false,
  teamId,
}) => {
  const [passThroughRoutes, setPassThroughRoutes] = useState<Array<{ label: string; value: string }>>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchPassThroughRoutes = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await getPassThroughEndpointsCall(accessToken, teamId);
        if (response.endpoints) {
          const routes = response.endpoints.flatMap((endpoint: PassThroughEndpoint) => {
            const path = endpoint.path;
            const methods = endpoint.methods;
            
            // If methods are specified, create one entry per method
            if (methods && methods.length > 0) {
              return methods.map((method) => ({
                label: `${method} ${path}`,
                value: path, // Keep value as path for backward compatibility
              }));
            }
            
            // If no methods specified, show just the path (all methods supported)
            return [{
              label: path,
              value: path,
            }];
          });
          setPassThroughRoutes(routes);
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
    <Select
      mode="tags"
      placeholder={placeholder}
      onChange={onChange}
      value={value}
      loading={loading}
      className={className}
      allowClear
      options={passThroughRoutes}
      optionFilterProp="label"
      showSearch
      style={{ width: "100%" }}
      disabled={disabled}
    />
  );
};

export default PassThroughRoutesSelector;

