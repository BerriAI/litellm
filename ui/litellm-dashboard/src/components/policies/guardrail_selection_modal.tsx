import React, { useState, useEffect } from "react";
import { Modal, Checkbox, Button, Divider, Tag, Select, Switch, Tooltip } from "antd";
import { CheckCircleOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { credentialListCall } from "../networking";

interface GuardrailInfo {
  guardrail_name: string;
  description: string;
  alreadyExists: boolean;
  definition: any;
}

interface PartnerGuardrailConfig {
  provider: string;
  label: string;
  description: string;
  credential_provider: string;
  provision_config: Record<string, any>;
}

export interface PartnerGuardrailSelection {
  provider: string;
  credential_name: string;
  provision_config: Record<string, any>;
  aws_region_name?: string;
}

interface GuardrailSelectionModalProps {
  visible: boolean;
  template: any;
  existingGuardrails: Set<string>;
  onConfirm: (
    selectedGuardrails: any[],
    partnerGuardrails?: PartnerGuardrailSelection[]
  ) => void;
  onCancel: () => void;
  isLoading?: boolean;
  progressInfo?: { current: number; total: number } | null;
  accessToken?: string;
}

const AWS_REGIONS = [
  "us-east-1",
  "us-east-2",
  "us-west-2",
  "eu-west-1",
  "eu-west-2",
  "eu-central-1",
  "ap-southeast-1",
  "ap-southeast-2",
  "ap-northeast-1",
];

const partnerProviderLogoMap: Record<string, string> = {
  bedrock: "../ui/assets/logos/bedrock.svg",
};

const ProviderLogo: React.FC<{ provider: string }> = ({ provider }) => {
  const logoSrc = partnerProviderLogoMap[provider];
  if (logoSrc) {
    return (
      <img
        src={logoSrc}
        alt={provider}
        style={{ height: "20px", width: "20px", objectFit: "contain" }}
        onError={(e) => { e.currentTarget.style.display = "none"; }}
      />
    );
  }
  return (
    <div className="w-5 h-5 rounded bg-gray-200 flex items-center justify-center text-xs font-bold text-gray-500">
      ?
    </div>
  );
};

const ProvisionConfigTags: React.FC<{ config: Record<string, any> }> = ({ config }) => (
  <div className="flex flex-wrap gap-1.5">
    {config.topicPolicyConfig && (
      <Tag color="orange" className="text-xs">
        {config.topicPolicyConfig.topicsConfig?.length || 0} topic policy(s)
      </Tag>
    )}
    {config.contentPolicyConfig && (
      <Tag color="red" className="text-xs">
        {config.contentPolicyConfig.filtersConfig?.length || 0} content filter(s)
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
      <Tag className="text-xs">{credential.credential_info.custom_llm_provider}</Tag>
    )}
  </div>
);

const PartnerGuardrailCard: React.FC<{
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
}> = ({
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
  const credentialOptions = (providerCredentials.length > 0 ? providerCredentials : allCredentials).map(
    (c: any) => ({
      value: c.credential_name,
      label: (
        <CredentialOption
          credential={c}
          provider={pg.provider}
          showProviderLogo={providerCredentials.length > 0}
        />
      ),
    })
  );

  return (
    <div
      className={`border rounded-lg p-4 transition-colors ${
        isEnabled ? "bg-orange-50 border-orange-200" : "bg-gray-50 border-gray-200"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 pt-0.5">
          <Switch size="small" checked={isEnabled} onChange={onToggle} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <ProviderLogo provider={pg.provider} />
            <span className="text-sm font-medium text-gray-900">{pg.label}</span>
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
                  placeholder={credentialsLoading ? "Loading credentials..." : "Select a credential"}
                  loading={credentialsLoading}
                  value={selectedCredential}
                  onChange={onCredentialChange}
                  options={credentialOptions}
                  notFoundContent={
                    <div className="text-center py-2 text-gray-500 text-xs">
                      No credentials found. Add one in Settings &rarr; Credentials.
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

const GuardrailSelectionModal: React.FC<GuardrailSelectionModalProps> = ({
  visible,
  template,
  existingGuardrails,
  onConfirm,
  onCancel,
  isLoading = false,
  progressInfo,
  accessToken,
}) => {
  const [selectedGuardrails, setSelectedGuardrails] = useState<Set<string>>(
    new Set()
  );

  // Partner guardrail state
  const [partnerEnabled, setPartnerEnabled] = useState<Record<string, boolean>>(
    {}
  );
  const [partnerCredentials, setPartnerCredentials] = useState<
    Record<string, string>
  >({});
  const [partnerRegions, setPartnerRegions] = useState<Record<string, string>>(
    {}
  );
  const [credentials, setCredentials] = useState<any[]>([]);
  const [credentialsLoading, setCredentialsLoading] = useState(false);

  const partnerGuardrails: PartnerGuardrailConfig[] =
    template?.partnerGuardrails || [];

  // Prepare guardrail info with existence status
  const guardrailsInfo: GuardrailInfo[] = (
    template?.guardrailDefinitions || []
  ).map((def: any) => ({
    guardrail_name: def.guardrail_name,
    description: def.guardrail_info?.description || "No description available",
    alreadyExists: existingGuardrails.has(def.guardrail_name),
    definition: def,
  }));

  // Initialize selection: select only new guardrails by default
  useEffect(() => {
    if (visible && template) {
      const newGuardrails = guardrailsInfo
        .filter((g) => !g.alreadyExists)
        .map((g) => g.guardrail_name);
      setSelectedGuardrails(new Set(newGuardrails));

      // Reset partner state
      setPartnerEnabled({});
      setPartnerCredentials({});
      setPartnerRegions({});
    }
  }, [visible, template]);

  // Fetch credentials when a partner guardrail is enabled
  useEffect(() => {
    const anyEnabled = Object.values(partnerEnabled).some(Boolean);
    if (anyEnabled && accessToken && credentials.length === 0) {
      setCredentialsLoading(true);
      credentialListCall(accessToken)
        .then((data) => {
          setCredentials(data?.credentials || []);
        })
        .catch((err) => {
          console.error("Failed to fetch credentials:", err);
        })
        .finally(() => {
          setCredentialsLoading(false);
        });
    }
  }, [partnerEnabled, accessToken]);

  const handleToggle = (guardrailName: string) => {
    setSelectedGuardrails((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(guardrailName)) {
        newSet.delete(guardrailName);
      } else {
        newSet.add(guardrailName);
      }
      return newSet;
    });
  };

  const handleSelectAll = () => {
    const allNew = guardrailsInfo
      .filter((g) => !g.alreadyExists)
      .map((g) => g.guardrail_name);
    setSelectedGuardrails(new Set(allNew));
  };

  const handleDeselectAll = () => {
    setSelectedGuardrails(new Set());
  };

  const handleConfirm = () => {
    const selectedDefinitions = guardrailsInfo
      .filter((g) => selectedGuardrails.has(g.guardrail_name))
      .map((g) => g.definition);

    // Build partner guardrail selections
    const partnerSelections: PartnerGuardrailSelection[] = [];
    for (const pg of partnerGuardrails) {
      if (partnerEnabled[pg.provider] && partnerCredentials[pg.provider]) {
        partnerSelections.push({
          provider: pg.provider,
          credential_name: partnerCredentials[pg.provider],
          provision_config: pg.provision_config,
          aws_region_name: partnerRegions[pg.provider] || undefined,
        });
      }
    }

    onConfirm(
      selectedDefinitions,
      partnerSelections.length > 0 ? partnerSelections : undefined
    );
  };

  // Filter credentials by provider type
  const getCredentialsForProvider = (credentialProvider: string) => {
    return credentials.filter((c: any) => {
      const provider =
        c.credential_info?.custom_llm_provider?.toLowerCase() || "";
      return provider.includes(credentialProvider.toLowerCase());
    });
  };

  const newGuardrailsCount = guardrailsInfo.filter(
    (g) => !g.alreadyExists
  ).length;
  const existingCount = guardrailsInfo.filter((g) => g.alreadyExists).length;
  const selectedCount = selectedGuardrails.size;

  return (
    <Modal
      title={
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold mb-0">{template?.title}</h3>
            {progressInfo && (
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-600 border border-blue-100">
                Template {progressInfo.current} of {progressInfo.total}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 font-normal mt-1">
            Review and select guardrails to create for this template
          </p>
        </div>
      }
      open={visible}
      onCancel={onCancel}
      width={700}
      footer={[
        <Button key="cancel" onClick={onCancel} disabled={isLoading}>
          Cancel
        </Button>,
        <Button
          key="confirm"
          type="primary"
          onClick={handleConfirm}
          loading={isLoading}
          disabled={selectedCount === 0 && existingCount === 0}
        >
          {selectedCount > 0
            ? `Create ${selectedCount} Guardrail${selectedCount > 1 ? "s" : ""} & Use Template`
            : "Use Template"}
        </Button>,
      ]}
    >
      <div className="py-4">
        {/* Summary Stats */}
        <div className="flex items-center gap-4 mb-4 p-3 bg-blue-50 rounded-lg border border-blue-100">
          <InfoCircleOutlined className="text-blue-600 text-lg" />
          <div className="flex-1">
            <div className="text-sm">
              <span className="font-medium text-gray-900">
                {guardrailsInfo.length} total guardrails
              </span>
              <span className="text-gray-600 mx-2">•</span>
              <span className="text-green-600 font-medium">
                {newGuardrailsCount} new
              </span>
              {existingCount > 0 && (
                <>
                  <span className="text-gray-600 mx-2">•</span>
                  <span className="text-gray-600">
                    {existingCount} already exist
                  </span>
                </>
              )}
            </div>
          </div>
          {newGuardrailsCount > 0 && (
            <div className="flex gap-2">
              <Button size="small" onClick={handleSelectAll}>
                Select All New
              </Button>
              <Button size="small" onClick={handleDeselectAll}>
                Deselect All
              </Button>
            </div>
          )}
        </div>

        {/* Guardrails List */}
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {guardrailsInfo.map((guardrail) => (
            <div
              key={guardrail.guardrail_name}
              className={`border rounded-lg p-4 ${
                guardrail.alreadyExists
                  ? "bg-gray-50 border-gray-200"
                  : "bg-white border-gray-300 hover:border-blue-400"
              } transition-colors`}
            >
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 pt-0.5">
                  {guardrail.alreadyExists ? (
                    <CheckCircleOutlined className="text-green-600 text-lg" />
                  ) : (
                    <Checkbox
                      checked={selectedGuardrails.has(guardrail.guardrail_name)}
                      onChange={() => handleToggle(guardrail.guardrail_name)}
                    />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono text-sm font-medium text-gray-900">
                      {guardrail.guardrail_name}
                    </span>
                    {guardrail.alreadyExists && (
                      <Tag color="green" className="text-xs">
                        Already exists
                      </Tag>
                    )}
                  </div>
                  <p className="text-sm text-gray-600">
                    {guardrail.description}
                  </p>

                  {/* Show guardrail type and mode */}
                  <div className="flex gap-2 mt-2">
                    <Tag className="text-xs">
                      {guardrail.definition?.litellm_params?.guardrail || "unknown"}
                    </Tag>
                    <Tag className="text-xs" color="blue">
                      {guardrail.definition?.litellm_params?.mode || "unknown"}
                    </Tag>
                    {guardrail.definition?.litellm_params?.patterns && (
                      <Tag className="text-xs" color="purple">
                        {guardrail.definition.litellm_params.patterns.length} pattern(s)
                      </Tag>
                    )}
                    {guardrail.definition?.litellm_params?.categories && (
                      <Tag className="text-xs" color="orange">
                        {guardrail.definition.litellm_params.categories.length} category/categories
                      </Tag>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {guardrailsInfo.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <p>No guardrails defined for this template.</p>
            <p className="text-sm mt-2">
              This template will use existing guardrails in your system.
            </p>
          </div>
        )}

        {/* Discovered Competitors */}
        {template?.discoveredCompetitors?.length > 0 && (
          <>
            <Divider />
            <div className="p-3 bg-purple-50 rounded-lg border border-purple-100">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">✨</span>
                <span className="font-medium text-purple-900 text-sm">
                  AI-Discovered Competitors ({template.discoveredCompetitors.length})
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {template.discoveredCompetitors.map((name: string) => (
                  <Tag key={name} color="purple" className="text-xs">
                    {name}
                  </Tag>
                ))}
              </div>
              <p className="text-xs text-purple-600 mt-2">
                These competitor names will be automatically blocked by the competitor-name-blocker guardrail.
              </p>
            </div>
          </>
        )}

        {/* Partner Guardrails Section */}
        {partnerGuardrails.length > 0 && (
          <>
            <Divider />
            <div className="space-y-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Enhance with Partner Guardrail
                </span>
                <Tooltip title="Optional. Provisions a cloud-based guardrail for stronger ML-powered detection alongside the keyword-based rules above.">
                  <InfoCircleOutlined className="text-gray-400 text-xs cursor-help" />
                </Tooltip>
                <Tag color="default" className="text-xs ml-1">
                  Optional
                </Tag>
              </div>

              {partnerGuardrails.map((pg) => (
                <PartnerGuardrailCard
                  key={pg.provider}
                  pg={pg}
                  isEnabled={partnerEnabled[pg.provider] || false}
                  onToggle={(checked) =>
                    setPartnerEnabled((prev) => ({ ...prev, [pg.provider]: checked }))
                  }
                  selectedCredential={partnerCredentials[pg.provider] || undefined}
                  onCredentialChange={(value) =>
                    setPartnerCredentials((prev) => ({ ...prev, [pg.provider]: value }))
                  }
                  selectedRegion={partnerRegions[pg.provider] || undefined}
                  onRegionChange={(value) =>
                    setPartnerRegions((prev) => ({ ...prev, [pg.provider]: value }))
                  }
                  providerCredentials={getCredentialsForProvider(pg.credential_provider)}
                  allCredentials={credentials}
                  credentialsLoading={credentialsLoading}
                />
              ))}
            </div>
          </>
        )}

        <Divider />

        {/* Selected Summary */}
        <div className="text-sm text-gray-600">
          {selectedCount > 0 ? (
            <p>
              <span className="font-medium text-gray-900">{selectedCount}</span>{" "}
              guardrail{selectedCount > 1 ? "s" : ""} will be created
              {Object.values(partnerEnabled).some(Boolean) && (
                <span className="text-orange-600">
                  {" "}
                  + partner guardrail will be provisioned
                </span>
              )}
            </p>
          ) : existingCount > 0 ? (
            <p className="text-green-600">
              All guardrails already exist. You can proceed to use this template.
            </p>
          ) : (
            <p className="text-orange-600">
              Select at least one guardrail to create, or click &quot;Use Template&quot; to proceed without creating new guardrails.
            </p>
          )}
        </div>
      </div>
    </Modal>
  );
};

export default GuardrailSelectionModal;
