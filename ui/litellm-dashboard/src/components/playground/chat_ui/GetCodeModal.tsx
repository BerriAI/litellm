import { Button, Modal, Select } from "antd";
import { Text } from "@tremor/react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import NotificationsManager from "../../molecules/notifications_manager";

interface GetCodeModalProps {
  visible: boolean;
  onClose: () => void;
  generatedCode: string;
  selectedSdk: "openai" | "azure";
  onSdkChange: (sdk: "openai" | "azure") => void;
}

function GetCodeModal({
  visible,
  onClose,
  generatedCode,
  selectedSdk,
  onSdkChange,
}: GetCodeModalProps) {
  return (
    <Modal
      title="Generated Code"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={800}
    >
      <div className="flex justify-between items-end my-4">
        <div>
          <Text className="font-medium block mb-1 text-gray-700">SDK Type</Text>
          <Select
            value={selectedSdk}
            onChange={(value) => onSdkChange(value as "openai" | "azure")}
            style={{ width: 150 }}
            options={[
              { value: "openai", label: "OpenAI SDK" },
              { value: "azure", label: "Azure SDK" },
            ]}
          />
        </div>
        <Button
          onClick={() => {
            navigator.clipboard.writeText(generatedCode);
            NotificationsManager.success("Copied to clipboard!");
          }}
        >
          Copy to Clipboard
        </Button>
      </div>
      <SyntaxHighlighter
        language="python"
        style={coy as any}
        wrapLines={true}
        wrapLongLines={true}
        className="rounded-md"
        customStyle={{
          maxHeight: "60vh",
          overflowY: "auto",
        }}
      >
        {generatedCode}
      </SyntaxHighlighter>
    </Modal>
  );
}

export default GetCodeModal;
