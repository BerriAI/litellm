import React from 'react';
import { Typography, Collapse } from 'antd';

const { Text } = Typography;
const { Panel } = Collapse;

export interface MCPEvent {
  type: string;
  sequence_number?: number;
  output_index?: number;
  item_id?: string;
  item?: {
    id?: string;
    type?: string;
    server_label?: string;
    tools?: Array<{
      name: string;
      description: string;
      annotations?: {
        read_only?: boolean;
      };
      input_schema?: any;
    }>;
    name?: string;
    arguments?: string;
    output?: string;
  };
  delta?: string;
  arguments?: string;
  timestamp?: number;
}

interface MCPEventsDisplayProps {
  events: MCPEvent[];
  className?: string;
}

const MCPEventsDisplay: React.FC<MCPEventsDisplayProps> = ({ events, className }) => {
  console.log("MCPEventsDisplay: Received events:", events);
  
  if (!events || events.length === 0) {
    console.log("MCPEventsDisplay: No events, returning null");
    return null;
  }

  // Find the list tools event
  const toolsEvent = events.find(event => 
    event.type === 'response.output_item.done' && 
    event.item?.type === 'mcp_list_tools' && 
    event.item.tools && 
    event.item.tools.length > 0
  );

  // Find MCP call events
  const mcpCallEvents = events.filter(event => 
    event.type === 'response.output_item.done' && 
    event.item?.type === 'mcp_call'
  );

  console.log("MCPEventsDisplay: toolsEvent:", toolsEvent);
  console.log("MCPEventsDisplay: mcpCallEvents:", mcpCallEvents);

  if (!toolsEvent && mcpCallEvents.length === 0) {
    console.log("MCPEventsDisplay: No valid events found, returning null");
    return null;
  }


  return (
    <div className={`mcp-events-display ${className || ''}`}>
      <style jsx>{`
        .openai-mcp-tools {
          position: relative;
          margin: 0;
          padding: 0;
        }
        .openai-mcp-tools .ant-collapse {
          background: transparent !important;
          border: none !important;
        }
        .openai-mcp-tools .ant-collapse-item {
          border: none !important;
          background: transparent !important;
        }
        .openai-mcp-tools .ant-collapse-header {
          padding: 0 0 0 20px !important;
          background: transparent !important;
          border: none !important;
          font-size: 14px !important;
          color: #9ca3af !important;
          font-weight: 400 !important;
          line-height: 20px !important;
          min-height: 20px !important;
        }
        .openai-mcp-tools .ant-collapse-header:hover {
          background: transparent !important;
          color: #6b7280 !important;
        }
        .openai-mcp-tools .ant-collapse-content {
          border: none !important;
          background: transparent !important;
        }
        .openai-mcp-tools .ant-collapse-content-box {
          padding: 4px 0 0 20px !important;
        }
        .openai-mcp-tools .ant-collapse-expand-icon {
          position: absolute !important;
          left: 2px !important;
          top: 2px !important;
          color: #9ca3af !important;
          font-size: 10px !important;
          width: 16px !important;
          height: 16px !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
        }
        .openai-mcp-tools .ant-collapse-expand-icon:hover {
          color: #6b7280 !important;
        }
        .openai-vertical-line {
          position: absolute;
          left: 9px;
          top: 18px;
          bottom: 0;
          width: 0.5px;
          background-color: #f3f4f6;
          opacity: 0.8;
        }
        .tool-item {
          font-family: ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 13px;
          color: #4b5563;
          line-height: 18px;
          padding: 0;
          margin: 0;
          background: white;
          position: relative;
          z-index: 1;
        }
        .mcp-section {
          margin-bottom: 12px;
          background: white;
          position: relative;
          z-index: 1;
        }
        .mcp-section:last-child {
          margin-bottom: 0;
        }
        .mcp-section-header {
          font-size: 13px;
          color: #6b7280;
          font-weight: 500;
          margin-bottom: 4px;
        }
        .mcp-code-block {
          background: #f9fafb;
          border: 1px solid #f3f4f6;
          border-radius: 6px;
          padding: 8px;
          font-size: 12px;
        }
        .mcp-json {
          font-family: ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          color: #374151;
          margin: 0;
          white-space: pre-wrap;
          word-wrap: break-word;
        }
        .mcp-approved {
          display: flex;
          align-items: center;
          font-size: 13px;
          color: #6b7280;
        }
        .mcp-checkmark {
          color: #10b981;
          margin-right: 6px;
          font-weight: bold;
        }
        .mcp-response-content {
          font-size: 13px;
          color: #374151;
          line-height: 1.5;
          white-space: pre-wrap;
          font-family: ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }
      `}</style>
      <div className="openai-mcp-tools">
        <div className="openai-vertical-line"></div>
        <Collapse 
          ghost 
          size="small"
          expandIconPosition="start"
          defaultActiveKey={toolsEvent ? ['list-tools'] : mcpCallEvents.map((_, index) => `mcp-call-${index}`)}
        >
          {/* List Tools Panel */}
          {toolsEvent && (
            <Panel 
              header="List tools" 
              key="list-tools"
            >
              <div>
                {toolsEvent.item?.tools?.map((tool, index) => (
                  <div key={index} className="tool-item">
                    {tool.name}
                  </div>
                ))}
              </div>
            </Panel>
          )}
          
          {/* MCP Call Panels */}
          {mcpCallEvents.map((callEvent, index) => (
            <Panel 
              header={callEvent.item?.name || 'Tool call'}
              key={`mcp-call-${index}`}
            >
              <div>
                {/* Request section */}
                <div className="mcp-section">
                  <div className="mcp-section-header">Request</div>
                  <div className="mcp-code-block">
                    {callEvent.item?.arguments && (
                      <pre className="mcp-json">
                        {(() => {
                          try {
                            return JSON.stringify(JSON.parse(callEvent.item.arguments), null, 2);
                          } catch (e) {
                            return callEvent.item.arguments;
                          }
                        })()}
                      </pre>
                    )}
                  </div>
                </div>
                
                {/* Approved section */}
                <div className="mcp-section">
                  <div className="mcp-approved">
                    <span className="mcp-checkmark">✓</span> Approved
                  </div>
                </div>
                
                {/* Response section */}
                {callEvent.item?.output && (
                  <div className="mcp-section">
                    <div className="mcp-section-header">Response</div>
                    <div className="mcp-response-content">
                      {callEvent.item.output}
                    </div>
                  </div>
                )}
              </div>
            </Panel>
          ))}
        </Collapse>
      </div>
    </div>
  );
};

export default MCPEventsDisplay;
