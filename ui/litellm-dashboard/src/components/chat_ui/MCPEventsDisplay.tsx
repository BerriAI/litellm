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
  if (!events || events.length === 0) {
    return null;
  }

  // Only show the final list tools event with the actual tools
  const toolsEvent = events.find(event => 
    event.type === 'response.output_item.done' && 
    event.item?.type === 'mcp_list_tools' && 
    event.item.tools && 
    event.item.tools.length > 0
  );

  if (!toolsEvent) {
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
      `}</style>
      <div className="openai-mcp-tools">
        <div className="openai-vertical-line"></div>
        <Collapse 
          ghost 
          size="small"
          expandIconPosition="start"
          defaultActiveKey={['1']}
        >
          <Panel 
            header="List tools" 
            key="1"
          >
            <div>
              {toolsEvent.item?.tools?.map((tool, index) => (
                <div key={index} className="tool-item">
                  {tool.name}
                </div>
              ))}
            </div>
          </Panel>
        </Collapse>
      </div>
    </div>
  );
};

export default MCPEventsDisplay;
