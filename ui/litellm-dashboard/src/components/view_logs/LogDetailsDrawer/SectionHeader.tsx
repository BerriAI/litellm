/**
 * SectionHeader - Datadog-style header with icon, label, metrics, and copy
 */

import { Typography, Button, Tooltip } from 'antd';
import { 
  MessageOutlined, 
  CopyOutlined,
  DownOutlined,
  UpOutlined
} from '@ant-design/icons';

const { Text } = Typography;

interface SectionHeaderProps {
  type: 'input' | 'output';
  tokens?: number;
  cost?: number;
  onCopy: () => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export function SectionHeader({ type, tokens, cost, onCopy, isCollapsed, onToggleCollapse }: SectionHeaderProps) {
  return (
    <div
      onClick={onToggleCollapse}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 16px',
        borderBottom: isCollapsed ? 'none' : '1px solid #f0f0f0',
        background: '#fafafa',
        cursor: onToggleCollapse ? 'pointer' : 'default',
        transition: 'background 0.15s ease',
      }}
      onMouseEnter={(e) => {
        if (onToggleCollapse) {
          e.currentTarget.style.background = '#f5f5f5';
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = '#fafafa';
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {/* Collapse Arrow */}
        {onToggleCollapse && (
          <div style={{ display: 'flex', alignItems: 'center' }}>
            {isCollapsed ? (
              <DownOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
            ) : (
              <UpOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
            )}
          </div>
        )}

        {/* Icon + Label */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {type === 'input' ? (
            <MessageOutlined style={{ color: '#8c8c8c', fontSize: 14 }} />
          ) : (
            <span style={{ fontSize: 14, filter: 'grayscale(1)', opacity: 0.6 }}>✨</span>
          )}
          <Text style={{ fontWeight: 500, fontSize: 14 }}>
            {type === 'input' ? 'Input' : 'Output'}
          </Text>
        </div>

        {/* Tokens */}
        {tokens !== undefined && (
          <Text type="secondary" style={{ fontSize: 12 }}>
            Tokens: {tokens.toLocaleString()}
          </Text>
        )}

        {/* Cost */}
        {cost !== undefined && (
          <Text type="secondary" style={{ fontSize: 12 }}>
            Cost: ${cost.toFixed(6)}
          </Text>
        )}
      </div>

      {/* Copy Button */}
      <Tooltip title="Copy">
        <Button 
          type="text" 
          size="small" 
          icon={<CopyOutlined />} 
          onClick={(e) => {
            e.stopPropagation(); // Prevent triggering collapse
            onCopy();
          }} 
        />
      </Tooltip>
    </div>
  );
}
