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
  TabPanels,
  Metric,
  Col,
  Text,
  SelectItem,
  TextInput,
  Button,
} from "@tremor/react";



import { message, Select } from "antd";
import { modelAvailableCall } from "./networking";
import openai from "openai";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { Typography } from "antd";

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
  if (isLocal != true) {
    console.log = function() {};
  }
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = isLocal
    ? "http://localhost:4000"
    : window.location.origin;
  const client = new openai.OpenAI({
    apiKey: accessToken, // Replace with your OpenAI API key
    baseURL: proxyBaseUrl, // Replace with your OpenAI API base URL
    dangerouslyAllowBrowser: true, // using a temporary litellm proxy key
  });

  try {
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
  } catch (error) {
    message.error(`Error occurred while generating model response. Please try again. Error: ${error}`, 20);
  }
}


const ChatUI: React.FC<ChatUIProps> = ({
  accessToken,
  token,
  userRole,
  userID,
}) => {
  const [apiKey, setApiKey] = useState("");
  const [inputMessage, setInputMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(
    undefined
  );
  const [modelInfo, setModelInfo] = useState<any[]>([]);// Declare modelInfo at the component level

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }

    

    // Fetch model info and set the default selected model
    const fetchModelInfo = async () => {
      try {
        const fetchedAvailableModels = await modelAvailableCall(
          accessToken,
          userID,
          userRole
        );
  
        console.log("model_info:", fetchedAvailableModels);
  
        if (fetchedAvailableModels?.data.length > 0) {
          const options = fetchedAvailableModels["data"].map((item: { id: string }) => ({
            value: item.id,
            label: item.id
          }));
  
          // Now, 'options' contains the list you wanted
          console.log(options); // You can log it to verify the list

          // if options.length > 0, only store unique values
          if (options.length > 0) {
            const uniqueModels = Array.from(new Set(options));

            console.log("Unique models:", uniqueModels);

            // sort uniqueModels alphabetically
            uniqueModels.sort((a: any, b: any) => a.label.localeCompare(b.label));


            console.log("Model info:", modelInfo);
            
            // setModelInfo(options) should be inside the if block to avoid setting it when no data is available
            setModelInfo(uniqueModels);
          }


          setSelectedModel(fetchedAvailableModels.data[0].id);
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
        // Handle error as needed
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

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleSendMessage();
    }
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() === "") return;

    if (!apiKey || !token || !userRole || !userID) {
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
          apiKey
        );
      }
    } catch (error) {
      console.error("Error fetching model response", error);
      updateUI("assistant", "Error fetching model response");
    }

    setInputMessage("");
  };

  if (userRole && userRole == "Admin Viewer") {
    const { Title, Paragraph } = Typography;
    return (
      <div>
        <Title level={1}>Access Denied</Title>
        <Paragraph>Ask your proxy admin for access to test models</Paragraph>
      </div>
    );
  }

  const onChange = (value: string) => {
    console.log(`selected ${value}`);
    setSelectedModel(value);
  };

  return (
    <div style={{ width: "100%", position: "relative" }}>
      <Grid className="gap-2 p-8 h-[80vh] w-full mt-2">
        <Card>
          
          <TabGroup>
            <TabList>
              <Tab>Chat</Tab>
            </TabList>

            <TabPanels>
              <TabPanel>
              <div className="sm:max-w-2xl">
          <Grid numItems={2}>
            <Col>
            <Text>API Key</Text>
              <TextInput placeholder="Type API Key here" type="password" onValueChange={setApiKey} value={apiKey}/>
            </Col>
            <Col className="mx-2">
            <Text>Select Model:</Text>

            <Select
                placeholder="Select a Model"
                onChange={onChange}
                options={modelInfo}
                style={{ width: "200px" }}
                
                
               
              />
            </Col>
          </Grid>
        
          
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
                        {/* <Title>Chat</Title> */}
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
                    <TextInput
                      type="text"
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyDown={handleKeyDown} // Add this line
                      placeholder="Type your message..."
                    />
                    <Button
                      onClick={handleSendMessage}
                      className="ml-2"
                    >
                      Send
                    </Button>
                  </div>
                </div>
              </TabPanel>
              
            </TabPanels>
          </TabGroup>
        </Card>
      </Grid>
    </div>
  );
};

export default ChatUI;
