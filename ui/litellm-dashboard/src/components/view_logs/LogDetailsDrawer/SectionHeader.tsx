/**
 * SectionHeader - Datadog-style header with icon, label, metrics, and copy
 */

import { Typography, Button, Tooltip } from 'antd';
import { 
  MessageOutlined, 
  ThunderboltOutlined, 
  CopyOutlined 
} from '@ant-design/icons';

const { Text } = Typography;

interface SectionHeaderProps {
  type: 'input' | 'output';
  tokens?: number;
  cost?: number;
  onCopy: () => void;
}

export function SectionHeader({ type, tokens, cost, onCopy }: SectionHeaderProps) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 14px',
        borderBottom: '1px solid #f0f0f0',
        background: '#fafafa',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        {/* Icon + Label */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {type === 'input' ? (
            <MessageOutlined style={{ color: '#8c8c8c', fontSize: 14 }} />
          ) : (
            <ThunderboltOutlined style={{ color: '#8c8c8c', fontSize: 14 }} />
          )}
          <Text strong style={{ fontSize: 13 }}>
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
        <Button type="text" size="small" icon={<CopyOutlined />} onClick={onCopy} />
      </Tooltip>
    </div>
  );
}
