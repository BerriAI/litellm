"use client";
import React, { useEffect, useState } from "react";
import {
  Badge,
  Card,
  Table,
  Metric,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
  Title,
  Icon,
  Accordion,
  AccordionBody,
  AccordionHeader,
  List,
  ListItem,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Grid,
} from "@tremor/react";
import { Statistic } from "antd"
import { modelAvailableCall }  from "./networking";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";


const APIRef = ({}) => {
    return (
        <>
         <Grid className="gap-2 p-8 h-[80vh] w-full mt-2">
        <div className="mb-5">
            <p className="text-2xl text-tremor-content-strong dark:text-dark-tremor-content-strong font-semibold">OpenAI Compatible Proxy: API Reference</p>        
            <Text className="mt-2 mb-2">LiteLLM is OpenAI Compatible. This means your API Key works with the OpenAI SDK. Just replace the base_url to point to your litellm proxy. Example Below </Text>

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
    base_url="http://0.0.0.0:4000" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo", # model to send to the proxy
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
    openai_api_base="http://0.0.0.0:4000",
    model = "gpt-3.5-turbo",
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

        
        </div>
        </Grid>

        
    </>
    )
}

export default APIRef;

