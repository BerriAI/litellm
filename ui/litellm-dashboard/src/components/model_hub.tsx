import React, { useEffect, useState } from 'react';

import { modelHubCall } from "./networking";

import { Card, Text, Title, Grid, Button, Badge, Tab,
    TabGroup,
    TabList,
    TabPanel,
    TabPanels, } from "@tremor/react";

import { RightOutlined, CopyOutlined } from '@ant-design/icons';

import { Modal, Tooltip } from 'antd';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";



interface ModelHubProps {

  userID: string;

  userRole: string;

  token: string;

  accessToken: string;

  keys: any; // Replace with the appropriate type for 'keys' prop

  premiumUser: boolean;

}



const ModelHub: React.FC<ModelHubProps> = ({

  userID,

  userRole,

  token,

  accessToken,

  keys,

  premiumUser,

}) => {



  const [modelHubData, setModelHubData] = useState(null);

  const [isModalVisible, setIsModalVisible] = useState(false);

  const [selectedModel, setSelectedModel] = useState(null);



  useEffect(() => {

    if (!accessToken || !token || !userRole || !userID) {

      return;

    }



    const fetchData = async () => {

      try {

        const _modelHubData = await modelHubCall(accessToken, userID, userRole);

        console.log("ModelHubData:", _modelHubData);

        setModelHubData(_modelHubData.data);

      } catch (error) {

        console.error("There was an error fetching the model data", error);

      }

    };



    fetchData();

  }, [accessToken, token, userRole, userID]);



  const showModal = (model) => {

    setSelectedModel(model);

    setIsModalVisible(true);

  };



  const handleOk = () => {

    setIsModalVisible(false);

    setSelectedModel(null);

  };



  const handleCancel = () => {

    setIsModalVisible(false);

    setSelectedModel(null);

  };



  const copyToClipboard = (text) => {

    navigator.clipboard.writeText(text);

  };



  return (

    <div>

<div className="w-full m-2 mt-2 p-8">

            <div className="relative w-full">

 

</div>


        <div className='flex items-center'>

        <Title className='ml-8 text-center '>Model Hub</Title>
        <Button className='ml-4'>
        <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
        ✨ Share
        </a>


</Button>
            
        </div>
      
              
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-4">

        
          {modelHubData && modelHubData.map((model: any) => (

            <Card

              key={model.model_group}

              className="mt-5 mx-8"

            >



              <pre className='flex justify-between'>
                

                <Title>{model.model_group}</Title>
                <Tooltip title={model.model_group}>

                    <CopyOutlined onClick={() => copyToClipboard(model.model_group)} style={{ cursor: 'pointer', marginRight: '10px' }} />

                    </Tooltip>

              </pre>

              <div className='my-5'>

              <Text>Mode: {model.mode}</Text>
              <Text>Supports Function Calling: {model?.supports_function_calling == true ? "Yes" : "No"}</Text>
              <Text>Supports Vision: {model?.supports_vision == true ? "Yes" : "No"}</Text>
              <Text>Max Input Tokens: {model?.max_input_tokens ? model?.max_input_tokens : "N/A"}</Text>
              <Text>Max Output Tokens: {model?.max_output_tokens ? model?.max_output_tokens : "N/A"}</Text>

              </div>

              <div style={{ marginTop: 'auto', textAlign: 'right' }}>

            

                <a href="#" onClick={() => showModal(model)} style={{ color: '#1890ff', fontSize: 'smaller' }}>

                  View more <RightOutlined />

                </a>

              </div>

            </Card>

          ))}

        </div>

      </div>

      <Modal

        title="Model Usage"
        width={800}

        visible={isModalVisible}
        footer={null}

        onOk={handleOk}

        onCancel={handleCancel}

      >

        {selectedModel && (

          <div>

            <p><strong>Model Name:</strong> {selectedModel.model_group}</p>
           
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