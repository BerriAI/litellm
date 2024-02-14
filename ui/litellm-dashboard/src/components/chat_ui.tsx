import React, { useState, useEffect } from "react";
import { Card, Title, Table, TableHead, TableRow, TableCell, TableBody, Grid } from "@tremor/react";
import { modelInfoCall } from "./networking";
import openai from "openai";



interface ChatUIProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
}

async function generateModelResponse(inputMessage: string, updateUI: (chunk: string) => void, selectedModel: string, accessToken: string) {
  const client = new openai.OpenAI({
    apiKey: accessToken, // Replace with your OpenAI API key
    baseURL: 'http://0.0.0.0:4000', // Replace with your OpenAI API base URL
    dangerouslyAllowBrowser: true, // using a temporary litellm proxy key
  });

  const response = await client.chat.completions.create({
    model: selectedModel,
    stream: true,
    messages: [
      {
        role: 'user',
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

const ChatUI: React.FC<ChatUIProps> = ({ accessToken, token, userRole, userID }) => {
    const [inputMessage, setInputMessage] = useState("");
    const [chatHistory, setChatHistory] = useState<any[]>([]);
    const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
    const [modelInfo, setModelInfo] = useState<any | null>(null); // Declare modelInfo at the component level
  
    useEffect(() => {
      // Fetch model info and set the default selected model
      const fetchModelInfo = async () => {
        const fetchedModelInfo = await modelInfoCall(accessToken, userID, userRole);
        console.log("model_info:", fetchedModelInfo);
  
        if (fetchedModelInfo?.data.length > 0) {
          setModelInfo(fetchedModelInfo);
          setSelectedModel(fetchedModelInfo.data[0].model_name);
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
  
      setChatHistory((prevHistory) => [
        ...prevHistory,
        { role: "user", content: inputMessage },
      ]);
  
      try {
        if (selectedModel) {
          await generateModelResponse(inputMessage, (chunk) => updateUI("assistant", chunk), selectedModel, accessToken);
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
          <div>
              <label>Select Model:</label>
              <select
                value={selectedModel || ""}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {/* Populate dropdown options from available models */}
                {modelInfo?.data.map((element: { model_name: string }) => (
                  <option key={element.model_name} value={element.model_name}>
                    {element.model_name}
                  </option>
                ))}
              </select>
            </div>
            <Table className="mt-5" style={{ display: "block", maxHeight: "60vh", overflowY: "auto" }}>
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
            <div className="mt-3" style={{ position: "absolute", bottom: 5, width: "95%" }}>
              <div className="flex">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  className="flex-1 p-2 border rounded-md mr-2"
                  placeholder="Type your message..."
                />
                <button onClick={handleSendMessage} className="p-2 bg-blue-500 text-white rounded-md">
                  Send
                </button>
              </div>
            </div>
          </Card>
        </Grid>
      </div>
    );
  };



  export default ChatUI;