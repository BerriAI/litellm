import React, { useState } from "react";
import { Typography, Space, Upload, Card, Button } from "antd";
import { PlusOutlined, UploadOutlined } from "@ant-design/icons";
import { validateBlockedWordsFile } from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";
import PatternModal from "./PatternModal";
import CustomPatternModal from "./CustomPatternModal";
import KeywordModal from "./KeywordModal";
import PatternTable from "./PatternTable";
import KeywordTable from "./KeywordTable";
import ContentCategoryConfiguration from "./ContentCategoryConfiguration";

const { Title, Text } = Typography;

interface PrebuiltPattern {
  name: string;
  display_name: string;
  category: string;
  description: string;
}

interface Pattern {
  id: string;
  type: "prebuilt" | "custom";
  name: string;
  display_name?: string;
  pattern?: string;
  action: "BLOCK" | "MASK";
}

interface BlockedWord {
  id: string;
  keyword: string;
  action: "BLOCK" | "MASK";
  description?: string;
}

interface ContentCategory {
  name: string;
  display_name: string;
  description: string;
  default_action: string;
}

interface SelectedContentCategory {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

interface ContentFilterConfigurationProps {
  prebuiltPatterns: PrebuiltPattern[];
  categories: string[];
  selectedPatterns: Pattern[];
  blockedWords: BlockedWord[];
  onPatternAdd: (pattern: Pattern) => void;
  onPatternRemove: (id: string) => void;
  onPatternActionChange: (id: string, action: "BLOCK" | "MASK") => void;
  onBlockedWordAdd: (word: BlockedWord) => void;
  onBlockedWordRemove: (id: string) => void;
  onBlockedWordUpdate: (id: string, field: string, value: any) => void;
  onFileUpload?: (content: string) => void;
  accessToken: string | null;
  showStep?: "patterns" | "keywords" | "categories";
  contentCategories?: ContentCategory[];
  selectedContentCategories?: SelectedContentCategory[];
  onContentCategoryAdd?: (category: SelectedContentCategory) => void;
  onContentCategoryRemove?: (id: string) => void;
  onContentCategoryUpdate?: (id: string, field: string, value: any) => void;
  pendingCategorySelection?: string;
  onPendingCategorySelectionChange?: (value: string) => void;
}

const ContentFilterConfiguration: React.FC<ContentFilterConfigurationProps> = ({
  prebuiltPatterns,
  categories,
  selectedPatterns,
  blockedWords,
  onPatternAdd,
  onPatternRemove,
  onPatternActionChange,
  onBlockedWordAdd,
  onBlockedWordRemove,
  onBlockedWordUpdate,
  onFileUpload,
  accessToken,
  showStep,
  contentCategories = [],
  selectedContentCategories = [],
  onContentCategoryAdd,
  onContentCategoryRemove,
  onContentCategoryUpdate,
  pendingCategorySelection,
  onPendingCategorySelectionChange,
}) => {
  const [patternModalVisible, setPatternModalVisible] = useState(false);
  const [keywordModalVisible, setKeywordModalVisible] = useState(false);
  const [customPatternModalVisible, setCustomPatternModalVisible] = useState(false);

  const [selectedPatternName, setSelectedPatternName] = useState<string>("");
  const [patternAction, setPatternAction] = useState<"BLOCK" | "MASK">("BLOCK");
  const [customPatternName, setCustomPatternName] = useState<string>("");
  const [customPatternRegex, setCustomPatternRegex] = useState<string>("");
  const [customPatternAction, setCustomPatternAction] = useState<"BLOCK" | "MASK">("BLOCK");
  const [newKeyword, setNewKeyword] = useState<string>("");
  const [newKeywordAction, setNewKeywordAction] = useState<"BLOCK" | "MASK">("BLOCK");
  const [newKeywordDescription, setNewKeywordDescription] = useState<string>("");
  const [uploadValidating, setUploadValidating] = useState(false);

  const handleAddPrebuiltPattern = () => {
    if (!selectedPatternName) {
      NotificationsManager.error("Please select a pattern");
      return;
    }

    const selectedPattern = prebuiltPatterns.find((p) => p.name === selectedPatternName);

    onPatternAdd({
      id: `pattern-${Date.now()}`,
      type: "prebuilt",
      name: selectedPatternName,
      display_name: selectedPattern?.display_name,
      action: patternAction,
    });

    setPatternModalVisible(false);
    setSelectedPatternName("");
    setPatternAction("BLOCK");
  };

  const handleAddCustomPattern = () => {
    if (!customPatternName || !customPatternRegex) {
      NotificationsManager.error("Please provide pattern name and regex");
      return;
    }

    onPatternAdd({
      id: `custom-${Date.now()}`,
      type: "custom",
      name: customPatternName,
      pattern: customPatternRegex,
      action: customPatternAction,
    });

    setCustomPatternModalVisible(false);
    setCustomPatternName("");
    setCustomPatternRegex("");
    setCustomPatternAction("BLOCK");
  };

  const handleAddKeyword = () => {
    if (!newKeyword) {
      NotificationsManager.error("Please enter a keyword");
      return;
    }

    onBlockedWordAdd({
      id: `word-${Date.now()}`,
      keyword: newKeyword,
      action: newKeywordAction,
      description: newKeywordDescription || undefined,
    });

    setKeywordModalVisible(false);
    setNewKeyword("");
    setNewKeywordDescription("");
    setNewKeywordAction("BLOCK");
  };

  const handleFileUpload = async (file: File) => {
    setUploadValidating(true);
    try {
      const content = await file.text();
      
      if (accessToken) {
        const result = await validateBlockedWordsFile(accessToken, content);
        if (result.valid) {
          if (onFileUpload) {
            onFileUpload(content);
          }
          NotificationsManager.success(result.message || "File uploaded successfully");
        } else {
          const errorMessage = result.error || (result.errors && result.errors.join(", ")) || "Invalid file";
          NotificationsManager.error(`Validation failed: ${errorMessage}`);
        }
      }
    } catch (error) {
      NotificationsManager.error(`Failed to upload file: ${error}`);
    } finally {
      setUploadValidating(false);
    }
    return false;
  };

  const showPatterns = !showStep || showStep === "patterns";
  const showKeywords = !showStep || showStep === "keywords";
  const showCategories = !showStep || showStep === "categories";

  return (
    <div className="space-y-6">
      {!showStep && (
        <div>
          <Text type="secondary">
            Configure patterns, keywords, and content categories to detect and filter sensitive information in requests and responses.
          </Text>
        </div>
      )}

      {showPatterns && (
        <Card
          title={
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Title level={5} style={{ margin: 0 }}>
                Pattern Detection
              </Title>
              <Text type="secondary" style={{ fontSize: 14, fontWeight: 400 }}>
                Detect sensitive information using regex patterns (SSN, credit cards, API keys, etc.)
              </Text>
            </div>
          }
          size="small"
        >
          <div style={{ marginBottom: 16 }}>
            <Space>
              <Button type="primary" onClick={() => setPatternModalVisible(true)} icon={<PlusOutlined />}>
                Add prebuilt pattern
              </Button>
              <Button onClick={() => setCustomPatternModalVisible(true)} icon={<PlusOutlined />}>
                Add custom regex
              </Button>
            </Space>
          </div>
          <PatternTable
            patterns={selectedPatterns}
            onActionChange={onPatternActionChange}
            onRemove={onPatternRemove}
          />
        </Card>
      )}

      {showKeywords && (
        <Card
          title={
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Title level={5} style={{ margin: 0 }}>
                Blocked Keywords
              </Title>
              <Text type="secondary" style={{ fontSize: 14, fontWeight: 400 }}>
                Block or mask specific sensitive terms and phrases
              </Text>
            </div>
          }
          size="small"
        >
          <div style={{ marginBottom: 16 }}>
            <Space>
              <Button type="primary" onClick={() => setKeywordModalVisible(true)} icon={<PlusOutlined />}>
                Add keyword
              </Button>
              <Upload beforeUpload={handleFileUpload} accept=".yaml,.yml" showUploadList={false}>
                <Button icon={<UploadOutlined />} loading={uploadValidating}>
                  Upload YAML file
                </Button>
              </Upload>
            </Space>
          </div>
          <KeywordTable
            keywords={blockedWords}
            onActionChange={onBlockedWordUpdate}
            onRemove={onBlockedWordRemove}
          />
        </Card>
      )}

      {showCategories && contentCategories.length > 0 && onContentCategoryAdd && onContentCategoryRemove && onContentCategoryUpdate && (
        <ContentCategoryConfiguration
          availableCategories={contentCategories}
          selectedCategories={selectedContentCategories}
          onCategoryAdd={onContentCategoryAdd}
          onCategoryRemove={onContentCategoryRemove}
          onCategoryUpdate={onContentCategoryUpdate}
          accessToken={accessToken}
          pendingSelection={pendingCategorySelection}
          onPendingSelectionChange={onPendingCategorySelectionChange}
        />
      )}

      <PatternModal
        visible={patternModalVisible}
        prebuiltPatterns={prebuiltPatterns}
        categories={categories}
        selectedPatternName={selectedPatternName}
        patternAction={patternAction}
        onPatternNameChange={setSelectedPatternName}
        onActionChange={(value) => setPatternAction(value as "BLOCK" | "MASK")}
        onAdd={handleAddPrebuiltPattern}
        onCancel={() => {
          setPatternModalVisible(false);
          setSelectedPatternName("");
          setPatternAction("BLOCK");
        }}
      />

      <CustomPatternModal
        visible={customPatternModalVisible}
        patternName={customPatternName}
        patternRegex={customPatternRegex}
        patternAction={customPatternAction}
        onNameChange={setCustomPatternName}
        onRegexChange={setCustomPatternRegex}
        onActionChange={(value) => setCustomPatternAction(value as "BLOCK" | "MASK")}
        onAdd={handleAddCustomPattern}
        onCancel={() => {
          setCustomPatternModalVisible(false);
          setCustomPatternName("");
          setCustomPatternRegex("");
          setCustomPatternAction("BLOCK");
        }}
      />

      <KeywordModal
        visible={keywordModalVisible}
        keyword={newKeyword}
        action={newKeywordAction}
        description={newKeywordDescription}
        onKeywordChange={setNewKeyword}
        onActionChange={(value) => setNewKeywordAction(value as "BLOCK" | "MASK")}
        onDescriptionChange={setNewKeywordDescription}
        onAdd={handleAddKeyword}
        onCancel={() => {
          setKeywordModalVisible(false);
          setNewKeyword("");
          setNewKeywordDescription("");
          setNewKeywordAction("BLOCK");
        }}
      />
    </div>
  );
};

export default ContentFilterConfiguration;
