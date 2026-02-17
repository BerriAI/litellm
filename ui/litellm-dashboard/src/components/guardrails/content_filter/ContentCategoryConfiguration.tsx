import React from "react";
import { Card, Typography, Select, Table, Tag, Collapse, Button } from "antd";
import { DeleteOutlined, PlusOutlined, FileTextOutlined } from "@ant-design/icons";
import { getCategoryYaml } from "../../networking";

const { Title, Text } = Typography;
const { Option } = Select;

interface ContentCategory {
  name: string;
  display_name: string;
  description: string;
  default_action: string;
}

interface SelectedCategory {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

interface ContentCategoryConfigurationProps {
  availableCategories: ContentCategory[];
  selectedCategories: SelectedCategory[];
  onCategoryAdd: (category: SelectedCategory) => void;
  onCategoryRemove: (id: string) => void;
  onCategoryUpdate: (id: string, field: string, value: any) => void;
  accessToken?: string | null;
  pendingSelection?: string;
  onPendingSelectionChange?: (value: string) => void;
}

const ContentCategoryConfiguration: React.FC<ContentCategoryConfigurationProps> = ({
  availableCategories,
  selectedCategories,
  onCategoryAdd,
  onCategoryRemove,
  onCategoryUpdate,
  accessToken,
  pendingSelection,
  onPendingSelectionChange,
}) => {
  // Use controlled state if parent provides it, otherwise use local state
  const [localSelectedCategoryName, setLocalSelectedCategoryName] = React.useState<string>("");
  const selectedCategoryName = pendingSelection !== undefined ? pendingSelection : localSelectedCategoryName;
  const setSelectedCategoryName = onPendingSelectionChange || setLocalSelectedCategoryName;
  const [categoryYaml, setCategoryYaml] = React.useState<{ [key: string]: string }>({});
  const [categoryFileTypes, setCategoryFileTypes] = React.useState<{ [key: string]: string }>({});
  const [loadingYaml, setLoadingYaml] = React.useState<{ [key: string]: boolean }>({});
  const [expandedYamlCategories, setExpandedYamlCategories] = React.useState<string[]>([]);
  const [previewYaml, setPreviewYaml] = React.useState<string>("");
  const [loadingPreviewYaml, setLoadingPreviewYaml] = React.useState<boolean>(false);

  const handleAddCategory = () => {
    if (!selectedCategoryName) {
      return;
    }

    const category = availableCategories.find((c) => c.name === selectedCategoryName);
    if (!category) {
      return;
    }

    // Check if already added
    if (selectedCategories.some((c) => c.category === selectedCategoryName)) {
      return;
    }

    onCategoryAdd({
      id: `category-${Date.now()}`,
      category: category.name,
      display_name: category.display_name,
      action: category.default_action as "BLOCK" | "MASK",
      severity_threshold: "medium",
    });

    setSelectedCategoryName("");
    setPreviewYaml(""); // Clear preview when category is added
  };

  const fetchCategoryYaml = async (categoryName: string) => {
    if (!accessToken) {
      return; // No access token
    }

    // Check if already loaded
    if (categoryYaml[categoryName]) {
      return;
    }

    setLoadingYaml((prev) => ({ ...prev, [categoryName]: true }));
    try {
      const data = await getCategoryYaml(accessToken, categoryName);
      let content = data.yaml_content;
      
      // Format JSON content for better readability
      if (data.file_type === 'json') {
        try {
          const parsed = JSON.parse(content);
          content = JSON.stringify(parsed, null, 2);
        } catch (e) {
          // If parsing fails, use original content
          console.warn(`Failed to format JSON for ${categoryName}:`, e);
        }
      }
      
      setCategoryYaml((prev) => ({ ...prev, [categoryName]: content }));
      setCategoryFileTypes((prev) => ({ ...prev, [categoryName]: data.file_type || 'yaml' }));
    } catch (error) {
      console.error(`Failed to fetch content for category ${categoryName}:`, error);
    } finally {
      setLoadingYaml((prev) => ({ ...prev, [categoryName]: false }));
    }
  };

  // Fetch preview YAML/JSON when a category is selected in dropdown
  React.useEffect(() => {
    if (selectedCategoryName && accessToken) {
      // Check if we already have this content cached
      const cachedContent = categoryYaml[selectedCategoryName];
      if (cachedContent) {
        setPreviewYaml(cachedContent);
        return;
      }

      // Fetch the content for preview
      setLoadingPreviewYaml(true);
      console.log(`Fetching content for category: ${selectedCategoryName}`, { accessToken: accessToken ? "present" : "missing" });
      getCategoryYaml(accessToken, selectedCategoryName)
        .then((data) => {
          console.log(`Successfully fetched content for ${selectedCategoryName}:`, data);
          let content = data.yaml_content;
          
          // Format JSON content for better readability
          if (data.file_type === 'json') {
            try {
              const parsed = JSON.parse(content);
              content = JSON.stringify(parsed, null, 2);
            } catch (e) {
              console.warn(`Failed to format JSON for ${selectedCategoryName}:`, e);
            }
          }
          
          setPreviewYaml(content);
          // Also cache it for later use
          setCategoryYaml((prev) => ({ ...prev, [selectedCategoryName]: content }));
          setCategoryFileTypes((prev) => ({ ...prev, [selectedCategoryName]: data.file_type || 'yaml' }));
        })
        .catch((error) => {
          console.error(`Failed to fetch preview content for category ${selectedCategoryName}:`, error);
          setPreviewYaml("");
        })
        .finally(() => {
          setLoadingPreviewYaml(false);
        });
    } else {
      setPreviewYaml("");
      setLoadingPreviewYaml(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCategoryName, accessToken]);

  const columns = [
    {
      title: "Category",
      dataIndex: "display_name",
      key: "display_name",
      render: (text: string, record: SelectedCategory) => {
        const category = availableCategories.find((c) => c.name === record.category);
        return (
          <div>
            <div style={{ fontWeight: 500 }}>{text}</div>
            {category?.description && (
              <div style={{ fontSize: "12px", color: "#888", marginTop: "4px" }}>
                {category.description}
              </div>
            )}
          </div>
        );
      },
    },
    {
      title: "Action",
      dataIndex: "action",
      key: "action",
      width: 150,
      render: (action: string, record: SelectedCategory) => (
        <Select
          value={action}
          onChange={(value) => onCategoryUpdate(record.id, "action", value)}
          style={{ width: "100%" }}
        >
          <Option value="BLOCK">
            <Tag color="red">BLOCK</Tag>
          </Option>
          <Option value="MASK">
            <Tag color="orange">MASK</Tag>
          </Option>
        </Select>
      ),
    },
    {
      title: "Severity Threshold",
      dataIndex: "severity_threshold",
      key: "severity_threshold",
      width: 180,
      render: (threshold: string, record: SelectedCategory) => (
        <Select
          value={threshold}
          onChange={(value) => onCategoryUpdate(record.id, "severity_threshold", value)}
          style={{ width: "100%" }}
        >
          <Option value="low">Low</Option>
          <Option value="medium">Medium</Option>
          <Option value="high">High</Option>
        </Select>
      ),
    },
    {
      title: "",
      key: "actions",
      width: 80,
      render: (_: any, record: SelectedCategory) => (
        <Button
          icon={<DeleteOutlined />}
          onClick={() => onCategoryRemove(record.id)}
          size="small"
        >
          Remove
        </Button>
      ),
    },
  ];

  const unselectedCategories = availableCategories.filter(
    (cat) => !selectedCategories.some((sel) => sel.category === cat.name)
  );

  return (
    <Card
      title={
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Title level={5} style={{ margin: 0 }}>
            Content Categories
          </Title>
          <Text type="secondary" style={{ fontSize: 14, fontWeight: 400 }}>
            Detect harmful content, bias, and inappropriate advice using semantic analysis
          </Text>
        </div>
      }
      size="small"
    >
      <div style={{ marginBottom: 16, display: "flex", gap: 8 }}>
        <Select
          placeholder="Select a content category"
          value={selectedCategoryName || undefined}
          onChange={setSelectedCategoryName}
          style={{ flex: 1 }}
          showSearch
          optionLabelProp="label"
          filterOption={(input, option) =>
            (option?.label?.toString().toLowerCase() ?? "").includes(input.toLowerCase())
          }
        >
          {unselectedCategories.map((cat) => (
            <Option key={cat.name} value={cat.name} label={cat.display_name}>
              <div>
                <div style={{ fontWeight: 500 }}>{cat.display_name}</div>
                <div style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}>
                  {cat.description}
                </div>
              </div>
            </Option>
          ))}
        </Select>
        <Button
          type="primary"
          onClick={handleAddCategory}
          disabled={!selectedCategoryName}
          icon={<PlusOutlined />}
        >
          Add
        </Button>
      </div>

      {/* Preview box - shown when category is selected but not yet added */}
      {selectedCategoryName && (
        <div
          style={{
            marginBottom: 16,
            padding: "12px",
            background: "#f9f9f9",
            border: "1px solid #e0e0e0",
            borderRadius: "4px",
          }}
        >
          <div style={{ marginBottom: 8, fontWeight: 500, fontSize: "14px" }}>
            Preview: {availableCategories.find((c) => c.name === selectedCategoryName)?.display_name}
            {categoryFileTypes[selectedCategoryName] && (
              <span style={{ marginLeft: 8, fontSize: "12px", color: "#888", fontWeight: 400 }}>
                ({categoryFileTypes[selectedCategoryName]?.toUpperCase()})
              </span>
            )}
          </div>
          {loadingPreviewYaml ? (
            <div style={{ padding: "16px", textAlign: "center", color: "#888" }}>
              Loading content...
            </div>
          ) : previewYaml ? (
            <pre
              style={{
                background: "#fff",
                padding: "12px",
                borderRadius: "4px",
                overflow: "auto",
                maxHeight: "300px",
                fontSize: "12px",
                lineHeight: "1.5",
                margin: 0,
                border: "1px solid #e0e0e0",
              }}
            >
              <code>{previewYaml}</code>
            </pre>
          ) : (
            <div style={{ padding: "8px", textAlign: "center", color: "#888", fontSize: "12px" }}>
              Unable to load category content
            </div>
          )}
        </div>
      )}

      {selectedCategories.length > 0 ? (
        <>
        <Table
          dataSource={selectedCategories}
          columns={columns}
          pagination={false}
          size="small"
          rowKey="id"
        />
          <div style={{ marginTop: 16 }}>
            <Collapse
              activeKey={expandedYamlCategories}
              onChange={(keys) => {
                const keyArray = Array.isArray(keys) ? keys : keys ? [keys] : [];
                const oldExpanded = new Set(expandedYamlCategories);
                
                // Find newly expanded categories and fetch their YAML
                keyArray.forEach((key) => {
                  const categoryName = key as string;
                  if (!oldExpanded.has(categoryName) && !categoryYaml[categoryName]) {
                    fetchCategoryYaml(categoryName);
                  }
                });
                
                setExpandedYamlCategories(keyArray as string[]);
              }}
              ghost
              items={selectedCategories.map((category) => {
                const fileType = categoryFileTypes[category.category] || 'yaml';
                const fileTypeLabel = fileType.toUpperCase();
                
                return {
                  key: category.category,
                  label: (
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <FileTextOutlined />
                      <span>View {fileTypeLabel} for {category.display_name}</span>
                    </div>
                  ),
                  children: loadingYaml[category.category] ? (
                    <div style={{ padding: "16px", textAlign: "center", color: "#888" }}>
                      Loading content...
                    </div>
                  ) : categoryYaml[category.category] ? (
                    <pre
                      style={{
                        background: "#f5f5f5",
                        padding: "16px",
                        borderRadius: "4px",
                        overflow: "auto",
                        maxHeight: "400px",
                        fontSize: "12px",
                        lineHeight: "1.5",
                        margin: 0,
                      }}
                    >
                      <code>{categoryYaml[category.category]}</code>
                    </pre>
                  ) : (
                    <div style={{ padding: "16px", textAlign: "center", color: "#888" }}>
                      Content will load when expanded
                    </div>
                  ),
                };
              })}
            />
          </div>
        </>
      ) : (
        <div
          style={{
            textAlign: "center",
            padding: "24px",
            color: "#888",
            border: "1px dashed #d9d9d9",
            borderRadius: "4px",
          }}
        >
          No content categories selected. Add categories to detect harmful content, bias, or inappropriate advice.
        </div>
      )}
    </Card>
  );
};

export default ContentCategoryConfiguration;

