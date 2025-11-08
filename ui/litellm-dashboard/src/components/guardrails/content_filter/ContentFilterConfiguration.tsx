import React, { useState } from "react";
import { Typography, Space, Upload, Card } from "antd";
import { PlusOutlined, UploadOutlined } from "@ant-design/icons";
import { Button } from "@tremor/react";
import { validateBlockedWordsFile } from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";
import PatternModal from "./PatternModal";
import CustomPatternModal from "./CustomPatternModal";
import KeywordModal from "./KeywordModal";
import PatternTable from "./PatternTable";
import KeywordTable from "./KeywordTable";

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
  showStep?: "patterns" | "keywords";
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

  return (
    <div className="space-y-6">
      {!showStep && (
        <div>
          <Text type="secondary">
            Configure patterns and keywords to detect and filter sensitive information in requests and responses.
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
              <Button type="button" onClick={() => setPatternModalVisible(true)} icon={PlusOutlined}>
                Add prebuilt pattern
              </Button>
              <Button type="button" onClick={() => setCustomPatternModalVisible(true)} variant="secondary" icon={PlusOutlined}>
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
              <Button type="button" onClick={() => setKeywordModalVisible(true)} icon={PlusOutlined}>
                Add keyword
              </Button>
              <Upload beforeUpload={handleFileUpload} accept=".yaml,.yml" showUploadList={false}>
                <Button type="button" variant="secondary" icon={UploadOutlined} loading={uploadValidating}>
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
