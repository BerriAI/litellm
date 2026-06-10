import React from "react";
import { Input } from "antd";
import { useTranslation } from "react-i18next";

interface routingStrategyArgs {
  ttl?: number;
  lowest_latency_buffer?: number;
}

const defaultLowestLatencyArgs: routingStrategyArgs = {
  ttl: 3600,
  lowest_latency_buffer: 0,
};

interface LatencyBasedConfigurationProps {
  routingStrategyArgs: { [key: string]: any };
}

const LatencyBasedConfiguration: React.FC<LatencyBasedConfigurationProps> = ({ routingStrategyArgs }) => {
  const { t } = useTranslation();

  const paramExplanation: { [key: string]: string } = {
    ttl: t("routerSettings.latencyBasedConfiguration.ttlDescription"),
    lowest_latency_buffer: t("routerSettings.latencyBasedConfiguration.lowestLatencyBufferDescription"),
  };

  return (
    <>
      <div className="space-y-6">
        <div className="max-w-3xl">
          <h3 className="text-sm font-medium text-gray-900">{t("routerSettings.latencyBasedConfiguration.title")}</h3>
          <p className="text-xs text-gray-500 mt-1">{t("routerSettings.latencyBasedConfiguration.subtitle")}</p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
          {Object.entries(routingStrategyArgs || defaultLowestLatencyArgs).map(([param, value]) => (
            <div key={param} className="space-y-2">
              <label className="block">
                <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">
                  {param.replace(/_/g, " ")}
                </span>
                <p className="text-xs text-gray-500 mt-0.5 mb-2">{paramExplanation[param] || ""}</p>
                <Input
                  name={param}
                  defaultValue={typeof value === "object" ? JSON.stringify(value, null, 2) : value?.toString()}
                  className="font-mono text-sm w-full"
                />
              </label>
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-gray-200" />
    </>
  );
};

export default LatencyBasedConfiguration;
