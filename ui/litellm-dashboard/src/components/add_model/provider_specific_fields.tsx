import React from "react";
import { Form, Select } from "antd";
import { TextInput, Text } from "@tremor/react";
import { Row, Col, Typography, Button as Button2, Upload, UploadProps } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import { provider_map, Providers } from "../provider_info_helpers";
import { CredentialItem } from "../networking";
const { Link } = Typography;


interface ProviderSpecificFieldsProps {
  selectedProvider: Providers;
  uploadProps?: UploadProps;
}

interface ProviderCredentialField {
  key: string;
  label: string;
  placeholder?: string;
  tooltip?: string;
  required?: boolean;
  type?: "text" | "password" | "select" | "upload";
  options?: string[];
  defaultValue?: string;
}

export interface CredentialValues {
  key: string;
  value: string;
}


export const createCredentialFromModel = (provider: string, modelData: any): CredentialItem => {
  console.log("provider", provider);
  console.log("modelData", modelData);
  const enumKey = Object.keys(provider_map).find(
    key => provider_map[key].toLowerCase() === provider.toLowerCase()
  );
  if (!enumKey) {
    throw new Error(`Provider ${provider} not found in provider_map`);
  }
  const providerEnum = Providers[enumKey as keyof typeof Providers];
  const providerFields = PROVIDER_CREDENTIAL_FIELDS[providerEnum] || [];
  const credentialValues: object = {};

  console.log("providerFields", providerFields);

  // Go through each field defined for this provider
  providerFields.forEach(field => {
    const value = modelData.litellm_params[field.key];
    console.log("field", field);
    console.log("value", value);
    if (value !== undefined) {
      (credentialValues as Record<string, string>)[field.key] = value.toString();
    }
  });

  const credential: CredentialItem = {
    credential_name: `${provider}-credential-${Math.floor(Math.random() * 1000000)}`,
    credential_values: credentialValues,
    credential_info: {
      custom_llm_provider: provider,
      description: `Credential for ${provider}. Created from model ${modelData.model_name}`,
    }
  }

  return credential;
};

const PROVIDER_CREDENTIAL_FIELDS: Record<Providers, ProviderCredentialField[]> = {
  [Providers.OpenAI]: [
    {
      key: "api_base",
      label: "API Base",
      type: "select",
      options: [
        "https://api.openai.com/v1",
        "https://eu.api.openai.com"
      ],
      defaultValue: "https://api.openai.com/v1"
    },
    {
      key: "organization",
      label: "OpenAI Organization ID",
      placeholder: "[OPTIONAL] my-unique-org"
    },
    {
      key: "api_key",
      label: "OpenAI API Key",
      type: "password",
      required: true
    }
  ],
  [Providers.OpenAI_Text]: [
    {
      key: "api_base",
      label: "API Base",
      type: "select",
      options: [
        "https://api.openai.com/v1",
        "https://eu.api.openai.com"
      ],
      defaultValue: "https://api.openai.com/v1"
    },
    {
      key: "organization",
      label: "OpenAI Organization ID",
      placeholder: "[OPTIONAL] my-unique-org"
    },
    {
      key: "api_key",
      label: "OpenAI API Key",
      type: "password",
      required: true
    }
  ],
  [Providers.Vertex_AI]: [
    {
      key: "vertex_project",
      label: "Vertex Project",
      placeholder: "adroit-cadet-1234..",
      required: true
    },
    {
      key: "vertex_location",
      label: "Vertex Location",
      placeholder: "us-east-1",
      required: true
    },
    {
      key: "vertex_credentials",
      label: "Vertex Credentials",
      required: true,
      type: "upload"
    }
  ],
  [Providers.AssemblyAI]: [
    {
      key: "api_base",
      label: "API Base",
      type: "select",
      required: true,
      options: [
        "https://api.assemblyai.com",
        "https://api.eu.assemblyai.com"
      ]
    },
    {
      key: "api_key",
      label: "AssemblyAI API Key",
      type: "password",
      required: true
    }
  ],
  [Providers.Azure]: [
    {
      key: "api_base",
      label: "API Base",
      placeholder: "https://...",
      required: true
    },
    {
      key: "api_version",
      label: "API Version",
      placeholder: "2023-07-01-preview",
      tooltip: "By default litellm will use the latest version. If you want to use a different version, you can specify it here"
    },
    {
      key: "base_model",
      label: "Base Model",
      placeholder: "azure/gpt-3.5-turbo"
    },
    {
      key: "api_key",
      label: "Azure API Key",
      type: "password",
      required: true
    }
  ],
  [Providers.Azure_AI_Studio]: [
    {
      key: "api_base",
      label: "API Base",
      placeholder: "https://<test>.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
      tooltip: "Enter your full Target URI from Azure Foundry here. Example:  https://litellm8397336933.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
      required: true
    },
    {
      key: "api_key",
      label: "Azure API Key",
      type: "password",
      required: true
    }
  ],
  [Providers.OpenAI_Compatible]: [
    {
      key: "api_base",
      label: "API Base",
      placeholder: "https://...",
      required: true
    },
    {
      key: "api_key",
      label: "OpenAI API Key",
      type: "password",
      required: true
    }
  ],
  [Providers.OpenAI_Text_Compatible]: [
    {
      key: "api_base",
      label: "API Base",
      placeholder: "https://...",
      required: true
    },
    {
      key: "api_key",
      label: "OpenAI API Key",
      type: "password",
      required: true
    }
  ],
  [Providers.Bedrock]: [
    {
      key: "aws_access_key_id",
      label: "AWS Access Key ID",
      required: true,
      tooltip: "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
    },
    {
      key: "aws_secret_access_key",
      label: "AWS Secret Access Key",
      required: true,
      tooltip: "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
    },
    {
      key: "aws_region_name",
      label: "AWS Region Name",
      placeholder: "us-east-1",
      required: true,
      tooltip: "You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
    }
  ],
  [Providers.Ollama]: [], // No specific fields needed
  [Providers.Anthropic]: [{
    key: "api_key",
    label: "API Key",
    placeholder: "sk-",
    type: "password",
    required: true
  }],
  [Providers.Google_AI_Studio]: [{
    key: "api_key",
    label: "API Key",
    placeholder: "aig-",
    type: "password",
    required: true
  }],
  [Providers.Groq]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.MistralAI]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.Deepseek]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.Cohere]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.Databricks]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.xAI]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.Cerebras]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.Sambanova]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.Perplexity]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.TogetherAI]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.Openrouter]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }],
  [Providers.FireworksAI]: [{
    key: "api_key",
    label: "API Key",
    type: "password",
    required: true
  }]
};

const ProviderSpecificFields: React.FC<ProviderSpecificFieldsProps> = ({
  selectedProvider,
  uploadProps
}) => {
  const selectedProviderEnum = Providers[selectedProvider as keyof typeof Providers] as Providers;
  
  // Simply use the fields as defined in PROVIDER_CREDENTIAL_FIELDS
  const allFields = React.useMemo(() => {
    return PROVIDER_CREDENTIAL_FIELDS[selectedProviderEnum] || [];
  }, [selectedProviderEnum]);

  return (
    <>
      {allFields.map((field) => (
        <React.Fragment key={field.key}>
          <Form.Item
            label={field.label}
            name={field.key}
            rules={field.required ? [{ required: true, message: "Required" }] : undefined}
            tooltip={field.tooltip}
            className={field.key === "vertex_credentials" ? "mb-0" : undefined}
          >
            {field.type === "select" ? (
              <Select 
                placeholder={field.placeholder}
                defaultValue={field.defaultValue}
              >
                {field.options?.map((option) => (
                  <Select.Option key={option} value={option}>
                    {option}
                  </Select.Option>
                ))}
              </Select>
            ) : field.type === "upload" ? (
              <Upload {...uploadProps}>
                <Button2 icon={<UploadOutlined />}>Click to Upload</Button2>
              </Upload>
            ) : (
              <TextInput 
                placeholder={field.placeholder} 
                type={field.type === "password" ? "password" : "text"} 
              />
            )}
          </Form.Item>

          {/* Special case for Vertex Credentials help text */}
          {field.key === "vertex_credentials" && (
            <Row>
              <Col span={10}></Col>
              <Col span={10}>
                <Text className="mb-3 mt-1">
                  Give litellm a gcp service account(.json file), so it
                  can make the relevant calls
                </Text>
              </Col>
            </Row>
          )}

          {/* Special case for Azure Base Model help text */}
          {field.key === "base_model" && (
            <Row>
              <Col span={10}></Col>
              <Col span={10}>
                <Text className="mb-2">
                  The actual model your azure deployment uses. Used
                  for accurate cost tracking. Select name from{" "}
                  <Link
                    href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                    target="_blank"
                  >
                    here
                  </Link>
                </Text>
              </Col>
            </Row>
          )}
        </React.Fragment>
      ))}
    </>
  );
};

export default ProviderSpecificFields;