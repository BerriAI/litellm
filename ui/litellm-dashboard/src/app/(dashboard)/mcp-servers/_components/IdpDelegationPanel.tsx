import React, { useCallback, useEffect, useState } from "react";
import { Button, Card, List, Space, Tag, Typography } from "antd";
import { CheckCircle2, Link as LinkIcon, RefreshCw } from "lucide-react";
import { getMcpIdpGrantsCall, getMcpIdpProvidersCall, getProxyBaseUrl } from "@/components/networking";

const { Title, Text } = Typography;

interface IdpProvider {
  token_url: string;
  label: string;
  grant_key: string;
}

interface IdpGrant {
  grant_key: string;
  connected_at: string | null;
  expires_at: string | null;
  offline_access: boolean;
}

interface IdpDelegationPanelProps {
  accessToken: string;
}

const describeGrant = (grant: IdpGrant): string => {
  const parts: string[] = [];
  if (grant.connected_at) {
    parts.push(`connected ${new Date(grant.connected_at).toLocaleString()}`);
  }
  if (grant.expires_at) {
    parts.push(`access token expires ${new Date(grant.expires_at).toLocaleString()}`);
  }
  return parts.length > 0 ? parts.join("; ") : "connected";
};

const IdpDelegationPanel: React.FC<IdpDelegationPanelProps> = ({ accessToken }) => {
  const proxyBaseUrl = getProxyBaseUrl();
  const [providers, setProviders] = useState<IdpProvider[] | null>(null);
  const [grantsByKey, setGrantsByKey] = useState<Record<string, IdpGrant>>({});

  const loadGrants = useCallback(async () => {
    try {
      const res = await getMcpIdpGrantsCall(accessToken);
      const grants: IdpGrant[] = res?.grants ?? [];
      setGrantsByKey(Object.fromEntries(grants.map((grant) => [grant.grant_key, grant])));
    } catch (error) {
      console.error("Failed to refresh MCP IdP grants:", error);
    }
  }, [accessToken]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await getMcpIdpProvidersCall(accessToken);
        if (!cancelled) {
          setProviders(res?.providers ?? []);
        }
        await loadGrants();
      } catch (error) {
        console.error("Failed to load MCP IdP providers:", error);
        if (!cancelled) {
          setProviders([]);
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [accessToken, loadGrants]);

  useEffect(() => {
    const onFocus = () => {
      void loadGrants();
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [loadGrants]);

  if (!providers || providers.length === 0) {
    return null;
  }

  const connect = (provider: IdpProvider) => {
    const url = `${proxyBaseUrl}/v1/mcp/idp/authorize?token_url=${encodeURIComponent(provider.token_url)}`;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  return (
    <Card className="mt-6 border border-gray-200">
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <div className="flex items-start justify-between">
          <div>
            <Title level={5} style={{ marginBottom: 4 }}>
              Delegated access
            </Title>
            <Text type="secondary">
              Connect your identity provider so agents you have authorized can act on your behalf when they call MCP
              tools
            </Text>
          </div>
          <Button icon={<RefreshCw size={14} />} onClick={() => void loadGrants()}>
            Refresh
          </Button>
        </div>
        <List
          dataSource={providers}
          renderItem={(provider) => {
            const grant = grantsByKey[provider.grant_key];
            const connected = Boolean(grant);
            return (
              <List.Item
                actions={[
                  <Button
                    key="connect"
                    type={connected ? "default" : "primary"}
                    icon={<LinkIcon size={14} />}
                    onClick={() => connect(provider)}
                  >
                    {connected ? "Reconnect" : "Connect"}
                  </Button>,
                ]}
              >
                <List.Item.Meta
                  title={
                    <Space>
                      <span>{provider.label}</span>
                      {connected && (
                        <Tag color="green" icon={<CheckCircle2 size={12} className="inline align-text-bottom" />}>
                          Connected
                        </Tag>
                      )}
                      {connected && !grant.offline_access && <Tag color="orange">no refresh token</Tag>}
                    </Space>
                  }
                  description={connected ? describeGrant(grant) : "not connected"}
                />
              </List.Item>
            );
          }}
        />
      </Space>
    </Card>
  );
};

export default IdpDelegationPanel;
