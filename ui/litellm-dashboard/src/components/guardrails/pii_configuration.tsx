import React from 'react';
import { Typography, Select, Tag, Divider } from 'antd';

const { Title, Text } = Typography;
const { Option } = Select;

interface PiiConfigurationProps {
  entities: string[];
  actions: string[];
  selectedEntities: string[];
  selectedActions: {[key: string]: string};
  onEntitySelect: (entity: string) => void;
  onActionSelect: (entity: string, action: string) => void;
}

/**
 * A reusable component for rendering PII entity selection and action configuration
 * Used in both add and edit guardrail forms
 */
const PiiConfiguration: React.FC<PiiConfigurationProps> = ({
  entities,
  actions,
  selectedEntities,
  selectedActions,
  onEntitySelect,
  onActionSelect
}) => {
  return (
    <>
      <Divider>PII Configuration</Divider>
      <div className="mb-4">
        <Text>Select PII types to detect and actions to take:</Text>
      </div>
      
      <div className="mb-4">
        <div className="flex flex-wrap gap-2 mb-4">
          {entities.map(entity => (
            <Tag
              key={entity}
              color={selectedEntities.includes(entity) ? 'blue' : 'default'}
              className="cursor-pointer text-sm py-1 px-2"
              onClick={() => onEntitySelect(entity)}
            >
              {entity}
            </Tag>
          ))}
        </div>
      </div>
      
      {selectedEntities.length > 0 && (
        <div className="mb-4">
          <Title level={5}>Configure Actions</Title>
          {selectedEntities.map(entity => (
            <div key={entity} className="flex items-center justify-between mb-2">
              <Text>{entity}</Text>
              <Select
                value={selectedActions[entity] || 'MASK'}
                onChange={(value) => onActionSelect(entity, value)}
                style={{ width: 120 }}
              >
                {actions.map(action => (
                  <Option key={action} value={action}>{action}</Option>
                ))}
              </Select>
            </div>
          ))}
        </div>
      )}
    </>
  );
};

export default PiiConfiguration; 