import React from "react";
import { Tag, Select, Switch } from "antd";
import {
  AWS_REGIONS,
  PartnerGuardrailConfig,
  partnerProviderLogoMap,
} from "./types";

export const ProviderLogo: React.FC<{ provider: string }> = ({ provider }) => {
  const logoSrc = partnerProviderLogoMap[provider];
  if (logoSrc) {
    return (
      <img
        src={logoSrc}
        alt={provider}
        style={{ height: "20px", width: "20px", objectFit: "contain" }}
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />
    );
  }
  return (
    <div className="w-5 h-5 rounded bg-gray-200 flex items-center justify-center text-xs font-bold text-gray-500">
      ?
    </div>
  );
};

const ProvisionConfigTags: React.FC<{ config: Record<string, any> }> = ({
  config,
}) => (
  <div className="flex flex-wrap gap-1.5">
    {config.topicPolicyConfig && (
      <Tag color="orange" className="text-xs">
        {config.topicPolicyConfig.topicsConfig?.length || 0} topic policy(s)
      </Tag>
    )}
    {config.contentPolicyConfig && (
      <Tag color="red" className="text-xs">
        {config.contentPolicyConfig.filtersConfig?.length || 0} content
        filter(s)
      </Tag>
    )}
    {config.wordPolicyConfig && (
      <Tag color="purple" className="text-xs">
        word policy
      </Tag>
    )}
    {config.sensitiveInformationPolicyConfig && (
      <Tag color="blue" className="text-xs">
        PII detection
      </Tag>
    )}
  </div>
);

const CredentialOption: React.FC<{
  credential: any;
  provider?: string;
  showProviderLogo?: boolean;
}> = ({ credential, provider, showProviderLogo }) => (
  <div className="flex items-center gap-2">
    {showProviderLogo && provider && <ProviderLogo provider={provider} />}
    <span>{credential.credential_name}</span>
    {!showProviderLogo && credential.credential_info?.custom_llm_provider && (
      <Tag className="text-xs">
        {credential.credential_info.custom_llm_provider}
      </Tag>
    )}
  </div>
);

export interface PartnerGuardrailCardProps {
  pg: PartnerGuardrailConfig;
  isEnabled: boolean;
  onToggle: (checked: boolean) => void;
  selectedCredential?: string;
  onCredentialChange: (value: string) => void;
  selectedRegion?: string;
  onRegionChange: (value: string) => void;
  providerCredentials: any[];
  allCredentials: any[];
  credentialsLoading: boolean;
}

const PartnerGuardrailCard: React.FC<PartnerGuardrailCardProps> = ({
  pg,
  isEnabled,
  onToggle,
  selectedCredential,
  onCredentialChange,
  selectedRegion,
  onRegionChange,
  providerCredentials,
  allCredentials,
  credentialsLoading,
}) => {
  const credentialOptions = (
    providerCredentials.length > 0 ? providerCredentials : allCredentials
  ).map((c: any) => ({
    value: c.credential_name,
    label: (
      <CredentialOption
        credential={c}
        provider={pg.provider}
        showProviderLogo={providerCredentials.length > 0}
      />
    ),
  }));

  return (
    <div
      className={`border rounded-lg p-4 transition-colors ${
        isEnabled
          ? "bg-orange-50 border-orange-200"
          : "bg-gray-50 border-gray-200"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 pt-0.5">
          <Switch size="small" checked={isEnabled} onChange={onToggle} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <ProviderLogo provider={pg.provider} />
            <span className="text-sm font-medium text-gray-900">
              {pg.label}
            </span>
          </div>
          <p className="text-xs text-gray-500 mb-3">{pg.description}</p>

          {isEnabled && (
            <div className="space-y-3 pt-2 border-t border-gray-200">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Credential
                </label>
                <Select
                  className="w-full"
                  size="small"
                  placeholder={
                    credentialsLoading
                      ? "Loading credentials..."
                      : "Select a credential"
                  }
                  loading={credentialsLoading}
                  value={selectedCredential}
                  onChange={onCredentialChange}
                  options={credentialOptions}
                  notFoundContent={
                    <div className="text-center py-2 text-gray-500 text-xs">
                      No credentials found. Add one in Settings &rarr;
                      Credentials.
                    </div>
                  }
                />
              </div>

              {pg.provider === "bedrock" && (
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    AWS Region
                  </label>
                  <Select
                    className="w-full"
                    size="small"
                    placeholder="Select region (default: us-east-1)"
                    value={selectedRegion}
                    onChange={onRegionChange}
                    options={AWS_REGIONS.map((r) => ({ value: r, label: r }))}
                    allowClear
                  />
                </div>
              )}

              <ProvisionConfigTags config={pg.provision_config} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PartnerGuardrailCard;
