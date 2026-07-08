import React, { useState, useEffect } from "react";
import {
  Card,
  Input,
  Select,
  Tabs,
  Tag,
  Typography,
  Button as AntButton,
  Modal,
  Spin,
  Alert,
  Space,
  Row,
  Col,
  Empty,
} from "antd";
import {
  AppstoreOutlined,
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  RobotOutlined,
} from "@ant-design/icons";
import { createAgentCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { Search } = Input;
const { Text, Title, Paragraph } = Typography;
const { TabPane } = Tabs;

interface XCTAgent {
  slug: string;
  name: string;
  description: string;
  category: string;
  emoji: string;
  system_prompt?: string;
}

interface XCTAgentMarketplaceProps {
  accessToken: string | null;
  isAdmin: boolean;
  gatewayUrl?: string;
  onAgentAdded?: () => void;
}

// Fallback used only if the server-side config endpoint is unreachable.
// Admins flip the live value via PUT /v1/xct-marketplace/config (S3-07).
const DEFAULT_GATEWAY_URL = "https://xct-agents-production.up.railway.app";

const XCTAgentMarketplace: React.FC<XCTAgentMarketplaceProps> = ({
  accessToken,
  isAdmin,
  gatewayUrl: gatewayUrlProp,
  onAgentAdded,
}) => {
  const [resolvedGatewayUrl, setResolvedGatewayUrl] = useState<string>(
    gatewayUrlProp || DEFAULT_GATEWAY_URL,
  );
  const gatewayUrl = resolvedGatewayUrl;

  // Resolve the marketplace gateway URL from the proxy (S3-07).
  useEffect(() => {
    if (gatewayUrlProp || !accessToken) {
      return;
    }
    let cancelled = false;
    fetch(`/v1/xct-marketplace/config`, {
      headers: { Authorization: `Bearer ${accessToken}` },
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((cfg) => {
        if (!cancelled && cfg && cfg.gateway_url) {
          setResolvedGatewayUrl(cfg.gateway_url);
        }
      })
      .catch(() => {
        /* fall back to DEFAULT_GATEWAY_URL */
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, gatewayUrlProp]);

  const [agents, setAgents] = useState<XCTAgent[]>([]);
  const [filteredAgents, setFilteredAgents] = useState<XCTAgent[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAgent, setSelectedAgent] = useState<XCTAgent | null>(null);
  const [agentDetail, setAgentDetail] = useState<XCTAgent | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isImporting, setIsImporting] = useState(false);

  // Fetch agents from gateway
  const fetchAgents = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${gatewayUrl}/agents`);
      if (!response.ok) {
        throw new Error(`Failed to fetch agents: ${response.statusText}`);
      }
      const data = await response.json();
      setAgents(data.agents || []);
      setFilteredAgents(data.agents || []);
    } catch (error) {
      console.error("Error fetching XCT agents:", error);
      NotificationsManager.error("Failed to fetch agents from XCT Agent Gateway");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAgents();
  }, [gatewayUrl]);

  // Filter agents based on category and search
  useEffect(() => {
    let filtered = agents;

    if (selectedCategory !== "all") {
      filtered = filtered.filter((agent) => agent.category === selectedCategory);
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (agent) =>
          agent.name.toLowerCase().includes(query) ||
          agent.description.toLowerCase().includes(query) ||
          agent.category.toLowerCase().includes(query)
      );
    }

    setFilteredAgents(filtered);
  }, [selectedCategory, searchQuery, agents]);

  // Get unique categories
  const categories = Array.from(new Set(agents.map((agent) => agent.category))).sort();

  // Fetch agent details
  const fetchAgentDetail = async (agent: XCTAgent) => {
    setSelectedAgent(agent);
    setLoadingDetail(true);
    setIsModalVisible(true);

    try {
      const response = await fetch(`${gatewayUrl}/agents/${agent.slug}/`);
      if (!response.ok) {
        throw new Error(`Failed to fetch agent details: ${response.statusText}`);
      }
      const data = await response.json();
      setAgentDetail({ ...agent, system_prompt: data.description });
    } catch (error) {
      console.error("Error fetching agent details:", error);
      NotificationsManager.error("Failed to fetch agent details");
    } finally {
      setLoadingDetail(false);
    }
  };

  // Import agent to LiteLLM
  const handleImportAgent = async () => {
    if (!selectedAgent || !accessToken) return;

    setIsImporting(true);
    try {
      // Fetch full agent details including system prompt
      const response = await fetch(`${gatewayUrl}/agents/${selectedAgent.slug}/`);
      if (!response.ok) {
        throw new Error(`Failed to fetch agent details: ${response.statusText}`);
      }
      const agentCard = await response.json();

      // Create agent in LiteLLM
      const agentData = {
        agent_name: `xct-${selectedAgent.slug}`,
        agent_type: "a2a",
        url: `${gatewayUrl}/agents/${selectedAgent.slug}/`,
        litellm_params: {
          model: "claude-3-5-sonnet-20241022", // Default model
        },
        agent_description: selectedAgent.description,
      };

      await createAgentCall(accessToken, agentData);

      NotificationsManager.success(
        `Agent "${selectedAgent.name}" imported successfully as "xct-${selectedAgent.slug}"`
      );
      setIsModalVisible(false);
      setSelectedAgent(null);
      setAgentDetail(null);

      if (onAgentAdded) {
        onAgentAdded();
      }
    } catch (error) {
      console.error("Error importing agent:", error);
      NotificationsManager.error(`Failed to import agent: ${error}`);
    } finally {
      setIsImporting(false);
    }
  };

  const handleModalClose = () => {
    setIsModalVisible(false);
    setSelectedAgent(null);
    setAgentDetail(null);
  };

  return (
    <div className="w-full">
      <Card>
        <Space direction="vertical" size="large" style={{ width: "100%" }}>
          <div>
            <div className="flex items-center gap-2 mb-4">
              <RobotOutlined style={{ fontSize: "24px" }} />
              <Title level={3} style={{ margin: 0 }}>
                XCT Agent Marketplace
              </Title>
            </div>
            <Paragraph>
              Browse and import 215+ specialized AI agents from the XCT Agent Gateway.
              Each agent is a pre-configured expert with domain-specific knowledge and personality.
            </Paragraph>
            <Alert
              message="About XCT Agents"
              description="These agents are served via the A2A protocol from the XCT Agent Gateway. When imported, they will appear in your agents list with the 'xct-' prefix."
              type="info"
              showIcon
              closable
              style={{ marginBottom: 16 }}
            />
          </div>

          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Search
                placeholder="Search agents by name, description, or category..."
                allowClear
                size="large"
                prefix={<SearchOutlined />}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{ width: "100%" }}
              />
            </Col>
            <Col span={8}>
              <Select
                size="large"
                style={{ width: "100%" }}
                value={selectedCategory}
                onChange={setSelectedCategory}
                options={[
                  { label: "All Categories", value: "all" },
                  ...categories.map((cat) => ({
                    label: cat.charAt(0).toUpperCase() + cat.slice(1).replace(/-/g, " "),
                    value: cat,
                  })),
                ]}
              />
            </Col>
            <Col span={4}>
              <AntButton
                type="default"
                icon={<ReloadOutlined />}
                onClick={fetchAgents}
                loading={loading}
                size="large"
                style={{ width: "100%" }}
              >
                Refresh
              </AntButton>
            </Col>
          </Row>

          {loading ? (
            <div style={{ textAlign: "center", padding: "40px" }}>
              <Spin size="large" tip="Loading agents..." />
            </div>
          ) : filteredAgents.length === 0 ? (
            <Empty
              description={
                searchQuery || selectedCategory !== "all"
                  ? "No agents found matching your filters"
                  : "No agents available"
              }
            />
          ) : (
            <div style={{ maxHeight: "600px", overflowY: "auto" }}>
              <Row gutter={[16, 16]}>
                {filteredAgents.map((agent) => (
                  <Col span={8} key={agent.slug}>
                    <Card
                      hoverable
                      size="small"
                      style={{ height: "100%" }}
                      onClick={() => fetchAgentDetail(agent)}
                    >
                      <Space direction="vertical" size="small" style={{ width: "100%" }}>
                        <div className="flex items-center gap-2">
                          <span style={{ fontSize: "20px" }}>{agent.emoji}</span>
                          <Text strong>{agent.name}</Text>
                        </div>
                        <Paragraph
                          ellipsis={{ rows: 3 }}
                          style={{ marginBottom: 8, fontSize: "12px" }}
                        >
                          {agent.description}
                        </Paragraph>
                        <Tag color="blue">
                          {agent.category.charAt(0).toUpperCase() +
                            agent.category.slice(1).replace(/-/g, " ")}
                        </Tag>
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </div>
          )}

          <div style={{ textAlign: "center", marginTop: 16 }}>
            <Text type="secondary">
              Showing {filteredAgents.length} of {agents.length} agents
            </Text>
          </div>
        </Space>
      </Card>

      <Modal
        title={
          <Space>
            <span style={{ fontSize: "20px" }}>{selectedAgent?.emoji}</span>
            <span>{selectedAgent?.name}</span>
          </Space>
        }
        open={isModalVisible}
        onCancel={handleModalClose}
        width={700}
        footer={[
          <AntButton key="close" onClick={handleModalClose}>
            Close
          </AntButton>,
          isAdmin && (
            <AntButton
              key="import"
              type="primary"
              icon={<PlusOutlined />}
              loading={isImporting}
              onClick={handleImportAgent}
            >
              Add to LiteLLM
            </AntButton>
          ),
        ]}
      >
        {loadingDetail ? (
          <div style={{ textAlign: "center", padding: "40px" }}>
            <Spin tip="Loading agent details..." />
          </div>
        ) : (
          <Space direction="vertical" size="middle" style={{ width: "100%" }}>
            <div>
              <Text strong>Category: </Text>
              <Tag color="blue">
                {selectedAgent?.category
                  ? selectedAgent.category.charAt(0).toUpperCase() +
                    selectedAgent.category.slice(1).replace(/-/g, " ")
                  : ""}
              </Tag>
            </div>
            <div>
              <Text strong>Agent ID: </Text>
              <Text code>xct-{selectedAgent?.slug}</Text>
            </div>
            <div>
              <Text strong>Description:</Text>
              <Paragraph style={{ marginTop: 8 }}>{selectedAgent?.description}</Paragraph>
            </div>
            {agentDetail?.system_prompt && (
              <div>
                <Text strong>System Prompt Preview:</Text>
                <Paragraph
                  style={{
                    marginTop: 8,
                    padding: 12,
                    background: "#f5f5f5",
                    borderRadius: 4,
                    maxHeight: 200,
                    overflow: "auto",
                  }}
                >
                  {agentDetail.system_prompt.substring(0, 500)}
                  {agentDetail.system_prompt.length > 500 && "..."}
                </Paragraph>
              </div>
            )}
            {!isAdmin && (
              <Alert
                message="Admin privileges required"
                description="Only administrators can import agents to LiteLLM."
                type="warning"
                showIcon
              />
            )}
          </Space>
        )}
      </Modal>
    </div>
  );
};

export default XCTAgentMarketplace;
