import React, { useState } from "react";
import { CodeIcon, CopyIcon } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const LANGUAGE_ITEMS = [
  { value: "curl", label: "cURL" },
  { value: "python", label: "Python (OpenAI SDK)" },
  { value: "javascript", label: "JavaScript (OpenAI SDK)" },
] as const;
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface PromptCodeSnippetsProps {
  promptId: string;
  model: string;
  promptVariables?: Record<string, string>;
  accessToken: string | null;
  version?: string;
  proxySettings?: {
    PROXY_BASE_URL?: string;
    LITELLM_UI_API_DOC_BASE_URL?: string | null;
  };
}

const PromptCodeSnippets: React.FC<PromptCodeSnippetsProps> = ({
  promptId,
  model,
  promptVariables = {},
  accessToken,
  version = "1",
  proxySettings,
}) => {
  const syntaxTheme = useSyntaxTheme(coy);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState<"curl" | "python" | "javascript">("curl");
  const [selectedTab, setSelectedTab] = useState("basic");
  const [generatedCode, setGeneratedCode] = useState("");

  const showModal = () => {
    setIsModalVisible(true);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
  };

  // Determine base URL with priority: LITELLM_UI_API_DOC_BASE_URL > PROXY_BASE_URL > window.location.origin
  let apiBase = window.location.origin;
  const customDocBaseUrl = proxySettings?.LITELLM_UI_API_DOC_BASE_URL;
  if (customDocBaseUrl && customDocBaseUrl.trim()) {
    apiBase = customDocBaseUrl;
  } else if (proxySettings?.PROXY_BASE_URL) {
    apiBase = proxySettings.PROXY_BASE_URL;
  }

  const effectiveApiKey = accessToken || "sk-1234";

  // Generate code based on selected language and tab
  const generateCode = () => {
    const hasVariables = Object.keys(promptVariables).length > 0;

    if (selectedLanguage === "curl") {
      if (selectedTab === "basic") {
        return `curl -X POST '${apiBase}/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${effectiveApiKey}' \\
  -d '{
    "model": "${model}",
    "prompt_id": "${promptId}"${
      hasVariables
        ? `,
    "prompt_variables": ${JSON.stringify(promptVariables, null, 6).replace(/\n/g, "\n    ")}`
        : ""
    }
  }' | jq`;
      } else if (selectedTab === "messages") {
        return `curl -X POST '${apiBase}/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${effectiveApiKey}' \\
  -d '{
    "model": "${model}",
    "prompt_id": "${promptId}"${
      hasVariables
        ? `,
    "prompt_variables": ${JSON.stringify(promptVariables, null, 6).replace(/\n/g, "\n    ")}`
        : ""
    },
    "messages": [
      {
        "role": "user",
        "content": "hi"
      }
    ]
  }' | jq`;
      } else {
        return `curl -X POST '${apiBase}/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${effectiveApiKey}' \\
  -d '{
    "model": "${model}",
    "prompt_id": "${promptId}",
    "prompt_version": ${version},
    "messages": [
      {
        "role": "user",
        "content": "Who are u"
      }
    ]
  }' | jq`;
      }
    } else if (selectedLanguage === "python") {
      const importCode = `import openai

client = openai.OpenAI(
    api_key="${effectiveApiKey}",
    base_url="${apiBase}"
)
`;
      if (selectedTab === "basic") {
        return `${importCode}
response = client.chat.completions.create(
    model="${model}",
    extra_body={
        "prompt_id": "${promptId}"${
          hasVariables
            ? `,
        "prompt_variables": ${JSON.stringify(promptVariables, null, 8).replace(/\n/g, "\n        ")}`
            : ""
        }
    }
)

print(response)`;
      } else if (selectedTab === "messages") {
        return `${importCode}
response = client.chat.completions.create(
    model="${model}",
    messages=[
        {"role": "user", "content": "hi"}
    ],
    extra_body={
        "prompt_id": "${promptId}"${
          hasVariables
            ? `,
        "prompt_variables": ${JSON.stringify(promptVariables, null, 8).replace(/\n/g, "\n        ")}`
            : ""
        }
    }
)

print(response)`;
      } else {
        return `${importCode}
response = client.chat.completions.create(
    model="${model}",
    messages=[
        {"role": "user", "content": "Who are u"}
    ],
    extra_body={
        "prompt_id": "${promptId}",
        "prompt_version": ${version}
    }
)

print(response)`;
      }
    } else {
      // JavaScript/Node.js
      const importCode = `import OpenAI from 'openai';

const client = new OpenAI({
    apiKey: "${effectiveApiKey}",
    baseURL: "${apiBase}"
});
`;
      if (selectedTab === "basic") {
        return `${importCode}
async function main() {
    const response = await client.chat.completions.create({
        model: "${model}",
        ${
          hasVariables
            ? `prompt_id: "${promptId}",
        prompt_variables: ${JSON.stringify(promptVariables, null, 8).replace(/\n/g, "\n        ")}`
            : `prompt_id: "${promptId}"`
        }
    });
    
    console.log(response);
}

main();`;
      } else if (selectedTab === "messages") {
        return `${importCode}
async function main() {
    const response = await client.chat.completions.create({
        model: "${model}",
        messages: [
            { role: "user", content: "hi" }
        ],
        ${
          hasVariables
            ? `prompt_id: "${promptId}",
        prompt_variables: ${JSON.stringify(promptVariables, null, 8).replace(/\n/g, "\n        ")}`
            : `prompt_id: "${promptId}"`
        }
    });
    
    console.log(response);
}

main();`;
      } else {
        return `${importCode}
async function main() {
    const response = await client.chat.completions.create({
        model: "${model}",
        messages: [
            { role: "user", content: "Who are u" }
        ],
        prompt_id: "${promptId}",
        prompt_version: ${version}
    });
    
    console.log(response);
}

main();`;
      }
    }
  };

  // Update generated code when language, tab or props change
  React.useEffect(() => {
    if (isModalVisible) {
      setGeneratedCode(generateCode());
    }
  }, [isModalVisible, selectedLanguage, selectedTab, promptId, model, promptVariables]);

  return (
    <>
      <Button variant="outline" onClick={showModal}>
        <CodeIcon />
        Get Code
      </Button>

      <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>Generated Code</DialogTitle>
          </DialogHeader>
          <div className="flex justify-between items-center mb-4">
            <div>
              <label htmlFor="prompt-code-language" className="font-medium block mb-1 text-foreground">
                Language
              </label>
              <Select
                items={LANGUAGE_ITEMS}
                value={selectedLanguage}
                onValueChange={(value) => setSelectedLanguage(value as "curl" | "python" | "javascript")}
              >
                <SelectTrigger id="prompt-code-language" className="w-[180px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LANGUAGE_ITEMS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button
              variant="outline"
              onClick={() => {
                navigator.clipboard.writeText(generatedCode);
                toast.success("Copied to clipboard!");
              }}
            >
              <CopyIcon />
              Copy to Clipboard
            </Button>
          </div>

          <Tabs value={selectedTab} onValueChange={(value) => setSelectedTab(String(value))}>
            <TabsList aria-label="Generated code type">
              <TabsTrigger value="basic">Basic</TabsTrigger>
              <TabsTrigger value="messages">With Messages</TabsTrigger>
              <TabsTrigger value="version">With Version</TabsTrigger>
            </TabsList>
          </Tabs>

          <SyntaxHighlighter
            language={selectedLanguage === "curl" ? "bash" : selectedLanguage === "python" ? "python" : "javascript"}
            style={syntaxTheme}
            wrapLines={true}
            wrapLongLines={true}
            className="rounded-md mt-0"
            customStyle={{
              maxHeight: "60vh",
              overflowY: "auto",
              marginTop: 0,
              borderTopLeftRadius: 0,
              borderTopRightRadius: 0,
            }}
          >
            {generatedCode}
          </SyntaxHighlighter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default PromptCodeSnippets;
