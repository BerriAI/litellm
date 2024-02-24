import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  Card,
  Title,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Grid,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  Metric,
  Select,
  SelectItem,
  TabPanels,
} from "@tremor/react";
import { modelAvailableCall } from "./networking";
import openai from "openai";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

interface ChatUIProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

async function generateModelResponse(
  inputMessage: string,
  updateUI: (chunk: string) => void,
  selectedModel: string,
  accessToken: string
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = isLocal
    ? "http://localhost:4000"
    : window.location.origin;
  const client = new openai.OpenAI({
    apiKey: accessToken, // Replace with your OpenAI API key
    baseURL: proxyBaseUrl, // Replace with your OpenAI API base URL
    dangerouslyAllowBrowser: true, // using a temporary litellm proxy key
  });

  const response = await client.chat.completions.create({
    model: selectedModel,
    stream: true,
    messages: [
      {
        role: "user",
        content: inputMessage,
      },
    ],
  });

  for await (const chunk of response) {
    console.log(chunk);
    if (chunk.choices[0].delta.content) {
      updateUI(chunk.choices[0].delta.content);
    }
  }
}

const ChatUI: React.FC<ChatUIProps> = ({
  accessToken,
  token,
  userRole,
  userID,
}) => {
  const [inputMessage, setInputMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(
    undefined
  );
  const [modelInfo, setModelInfo] = useState<any | null>(null); // Declare modelInfo at the component level

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    // Fetch model info and set the default selected model
    const fetchModelInfo = async () => {
      const fetchedAvailableModels = await modelAvailableCall(
        accessToken,
        userID,
        userRole
      );
      console.log("model_info:", fetchedAvailableModels);

      if (fetchedAvailableModels?.data.length > 0) {
        setModelInfo(fetchedAvailableModels.data);
        setSelectedModel(fetchedAvailableModels.data[0].id);
      }
    };

    fetchModelInfo();
  }, [accessToken, userID, userRole]);

  const updateUI = (role: string, chunk: string) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === role) {
        return [
          ...prevHistory.slice(0, prevHistory.length - 1),
          { role, content: lastMessage.content + chunk },
        ];
      } else {
        return [...prevHistory, { role, content: chunk }];
      }
    });
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() === "") return;

    if (!accessToken || !token || !userRole || !userID) {
      return;
    }

    setChatHistory((prevHistory) => [
      ...prevHistory,
      { role: "user", content: inputMessage },
    ]);

    try {
      if (selectedModel) {
        await generateModelResponse(
          inputMessage,
          (chunk) => updateUI("assistant", chunk),
          selectedModel,
          accessToken
        );
      }
    } catch (error) {
      console.error("Error fetching model response", error);
      updateUI("assistant", "Error fetching model response");
    }

    setInputMessage("");
  };

  return (
    <div style={{ width: "100%", position: "relative" }}>
      <Grid className="gap-2 p-10 h-[75vh] w-full">
        <Card>
          <TabGroup>
            <TabList className="mt-4">
              <Tab>Chat</Tab>
              <Tab>API Reference</Tab>
            </TabList>

            <TabPanels>
              <TabPanel>
                <div>
                  <label>Select Model:</label>
                  <select
                    value={selectedModel || ""}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {/* Populate dropdown options from available models */}
                    {modelInfo?.map((element: { id: string }) => (
                      <option key={element.id} value={element.id}>
                        {element.id}
                      </option>
                    ))}
                  </select>
                </div>
                <Table
                  className="mt-5"
                  style={{
                    display: "block",
                    maxHeight: "60vh",
                    overflowY: "auto",
                  }}
                >
                  <TableHead>
                    <TableRow>
                      <TableCell>
                        <Title>Chat</Title>
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {chatHistory.map((message, index) => (
                      <TableRow key={index}>
                        <TableCell>{`${message.role}: ${message.content}`}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <div
                  className="mt-3"
                  style={{ position: "absolute", bottom: 5, width: "95%" }}
                >
                  <div className="flex">
                    <input
                      type="text"
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      className="flex-1 p-2 border rounded-md mr-2"
                      placeholder="Type your message..."
                    />
                    <button
                      onClick={handleSendMessage}
                      className="p-2 bg-blue-500 text-white rounded-md"
                    >
                      Send
                    </button>
                  </div>
                </div>
              </TabPanel>
              <TabPanel>
                <TabGroup>
                  <TabList>
                    <Tab>OpenAI Python SDK</Tab>
                    <Tab>LlamaIndex</Tab>
                    <Tab>Langchain Py</Tab>
                  </TabList>
                  <TabPanels>
                    <TabPanel>
                      <SyntaxHighlighter language="python">
                        {`
import openai
client = openai.OpenAI(
    api_key="your_api_key",
    base_url="http://0.0.0.0:4000" # proxy base url
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo", # model to use from Models Tab
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={
        "metadata": {
            "generation_name": "ishaan-generation-openai-client",
            "generation_id": "openai-client-gen-id22",
            "trace_id": "openai-client-trace-id22",
            "trace_user_id": "openai-client-user-id2"
        }
    }
)

print(response)
            `}
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
    engine="azure-gpt-3.5",               # model_name on litellm proxy
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
    openai_api_base="http://0.0.0.0:8000",
    model = "gpt-3.5-turbo",
    temperature=0.1,
    extra_body={
        "metadata": {
            "generation_name": "ishaan-generation-langchain-client",
            "generation_id": "langchain-client-gen-id22",
            "trace_id": "langchain-client-trace-id22",
            "trace_user_id": "langchain-client-user-id2"
        }
    }
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
              </TabPanel>
            </TabPanels>
          </TabGroup>
        </Card>
      </Grid>
    </div>
  );
};

export default ChatUI;
