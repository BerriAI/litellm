import React, { useEffect, useState } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";

import { modelHubCall } from "./networking";
import { getConfigFieldSetting, updateConfigFieldSetting } from "./networking";
import {
  Card,
  Text,
  Title,
  Grid,
  Button,
  Badge,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
} from "@tremor/react";
import { RightOutlined, CopyOutlined } from "@ant-design/icons";

import { Modal, Tooltip, message } from "antd";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

interface ModelHubProps {
  accessToken: string | null;
  publicPage: boolean;
  premiumUser: boolean;
}

interface ModelInfo {
  model_group: string;
  mode: string;
  supports_function_calling: boolean;
  supports_vision: boolean;
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  supported_openai_params?: string[];
}

const ModelHub: React.FC<ModelHubProps> = ({
  accessToken,
  publicPage,
  premiumUser,
}) => {
  const [publicPageAllowed, setPublicPageAllowed] = useState<boolean>(false);
  const [modelHubData, setModelHubData] = useState<ModelInfo[] | null>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isPublicPageModalVisible, setIsPublicPageModalVisible] =
    useState(false);
  const [selectedModel, setSelectedModel] = useState<null | ModelInfo>(null);
  const router = useRouter();

  useEffect(() => {
    if (!accessToken) {
      return;
    }

    const fetchData = async () => {
      try {
        const _modelHubData = await modelHubCall(accessToken);

        console.log("ModelHubData:", _modelHubData);

        setModelHubData(_modelHubData.data);

        getConfigFieldSetting(accessToken, "enable_public_model_hub")
          .then((data) => {
            console.log(`data: ${JSON.stringify(data)}`);
            if (data.field_value == true) {
              setPublicPageAllowed(true);
            }
          })
          .catch((error) => {
            // do nothing
          });
      } catch (error) {
        console.error("There was an error fetching the model data", error);
      }
    };

    fetchData();
  }, [accessToken, publicPage]);

  const showModal = (model: ModelInfo) => {
    setSelectedModel(model);

    setIsModalVisible(true);
  };

  const goToPublicModelPage = () => {
    router.replace(`/model_hub?key=${accessToken}`);
  };
  const handleMakePublicPage = async () => {
    if (!accessToken) {
      return;
    }
    updateConfigFieldSetting(accessToken, "enable_public_model_hub", true).then(
      (data) => {
        setIsPublicPageModalVisible(true);
      }
    );
  };

  const handleOk = () => {
    setIsModalVisible(false);
    setIsPublicPageModalVisible(false);
    setSelectedModel(null);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setIsPublicPageModalVisible(false);
    setSelectedModel(null);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div>
      {(publicPage && publicPageAllowed) || publicPage == false ? (
        <div className="w-full m-2 mt-2 p-8">
          <div className="relative w-full"></div>

          <div
            className={`flex ${publicPage ? "justify-between" : "items-center"}`}
          >
            <Title className="ml-8 text-center ">Model Hub</Title>
            {publicPage == false ? (
              premiumUser ? (
                <Button className="ml-4" onClick={() => handleMakePublicPage()}>
                  ✨ Make Public
                </Button>
              ) : (
                <Button className="ml-4">
                  <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                    ✨ Make Public
                  </a>
                </Button>
              )
            ) : (
              <div className="flex justify-between items-center">
                <p>Filter by key:</p>
                <Text className="bg-gray-200 pr-2 pl-2 pt-1 pb-1 text-center">{`/ui/model_hub?key=<YOUR_KEY>`}</Text>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-4 pr-8">
            {modelHubData &&
              modelHubData.map((model: ModelInfo) => (
                <Card key={model.model_group} className="mt-5 mx-8">
                  <pre className="flex justify-between">
                    <Title>{model.model_group}</Title>
                    <Tooltip title={model.model_group}>
                      <CopyOutlined
                        onClick={() => copyToClipboard(model.model_group)}
                        style={{ cursor: "pointer", marginRight: "10px" }}
                      />
                    </Tooltip>
                  </pre>
                  <div className="my-5">
                    <Text>
                      Max Input Tokens:{" "}
                      {model?.max_input_tokens
                        ? model?.max_input_tokens
                        : "Unknown"}
                    </Text>
                    <Text>
                      Max Output Tokens:{" "}
                      {model?.max_output_tokens
                        ? model?.max_output_tokens
                        : "Unknown"}
                    </Text>
                    <Text>
                      Input Cost Per 1M Tokens (USD):{" "}
                      {model?.input_cost_per_token
                        ? `$${(model.input_cost_per_token * 1_000_000).toFixed(2)}`
                        : "Unknown"}
                    </Text>
                    <Text>
                      Output Cost Per 1M Tokens (USD):{" "}
                      {model?.output_cost_per_token
                        ? `$${(model.output_cost_per_token * 1_000_000).toFixed(2)}`
                        : "Unknown"}
                    </Text>
                  </div>
                  <div style={{ marginTop: "auto", textAlign: "right" }}>
                    <a
                      href="#"
                      onClick={() => showModal(model)}
                      style={{ color: "#1890ff", fontSize: "smaller" }}
                    >
                      View more <RightOutlined />
                    </a>
                  </div>
                </Card>
              ))}
          </div>
        </div>
      ) : (
        <Card className="mx-auto max-w-xl mt-10">
          <Text className="text-xl text-center mb-2 text-black">
            Public Model Hub not enabled.
          </Text>
          <p className="text-base text-center text-slate-800">
            Ask your proxy admin to enable this on their Admin UI.
          </p>
        </Card>
      )}

      <Modal
        title={"Public Model Hub"}
        width={600}
        visible={isPublicPageModalVisible}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <div className="pt-5 pb-5">
          <div className="flex justify-between mb-4">
            <Text className="text-base mr-2">Shareable Link:</Text>
            <Text className="max-w-sm ml-2 bg-gray-200 pr-2 pl-2 pt-1 pb-1 text-center rounded">{`<proxy_base_url>/ui/model_hub?key=<YOUR_API_KEY>`}</Text>
          </div>
          <div className="flex justify-end">
            <Button onClick={goToPublicModelPage}>See Page</Button>
          </div>
        </div>
      </Modal>
      <Modal
        title={
          selectedModel && selectedModel.model_group
            ? selectedModel.model_group
            : "Unknown Model"
        }
        width={800}
        visible={isModalVisible}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        {selectedModel && (
          <div>
            <p className="mb-4">
              <strong>Model Information & Usage</strong>
            </p>

            <TabGroup>
              <TabList>
                <Tab>Model Information</Tab>
                <Tab>OpenAI Python SDK</Tab>
                <Tab>Supported OpenAI Params</Tab>
                <Tab>LlamaIndex</Tab>
                <Tab>Langchain Py</Tab>
              </TabList>
              <TabPanels>
                <TabPanel>
                  <Text>
                    <strong>Model Group:</strong> 
                    <pre>{JSON.stringify(selectedModel, null, 2)}</pre>
                  </Text>
                </TabPanel>
                <TabPanel>
                  <SyntaxHighlighter language="python">
                    {`
import openai
client = openai.OpenAI(
    api_key="your_api_key",
    base_url="http://0.0.0.0:4000" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="${selectedModel.model_group}", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)
            `}
                  </SyntaxHighlighter>
                </TabPanel>
                <TabPanel>
                  <SyntaxHighlighter language="python">
                    {`${selectedModel.supported_openai_params?.map((param) => `${param}\n`).join("")}`}
                  </SyntaxHighlighter>
                </TabPanel>
                <TabPanel>
                  <SyntaxHighlighter language="python">
                    {`
import os, dotenv

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

llm = AzureOpenAI(
    engine="${selectedModel.model_group}",               # model_name on litellm proxy
    temperature=0.0,
    azure_endpoint="http://0.0.0.0:4000", # litellm proxy endpoint
    api_key="sk-1234",                    # litellm proxy API Key
    api_version="2023-07-01-preview",
)

embed_model = AzureOpenAIEmbedding(
    deployment_name="azure-embedding-model",
    azure_endpoint="http://0.0.0.0:4000",
    api_key="sk-1234",
    api_version="2023-07-01-preview",
)


documents = SimpleDirectoryReader("llama_index_data").load_data()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

            `}
                  </SyntaxHighlighter>
                </TabPanel>
                <TabPanel>
                  <SyntaxHighlighter language="python">
                    {`
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model = "${selectedModel.model_group}",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)

            `}
                  </SyntaxHighlighter>
                </TabPanel>
              </TabPanels>
            </TabGroup>

            {/* <p><strong>Additional Params:</strong> {JSON.stringify(selectedModel.litellm_params)}</p> */}

            {/* Add other model details here */}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default ModelHub;
