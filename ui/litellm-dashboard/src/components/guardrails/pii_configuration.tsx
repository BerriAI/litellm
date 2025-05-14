import React, { useState } from 'react';
import { Typography, Select, Button, Checkbox, Tooltip, Badge, Tag } from 'antd';
import { TextInput } from '@tremor/react';
import { SafetyOutlined, CloseOutlined, EyeInvisibleOutlined, StopOutlined, FilterOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;

interface PiiEntity {
  name: string;
  category: string;
}

interface PiiEntityCategory {
  category: string;
  entities: string[];
}

interface PiiConfigurationProps {
  entities: string[];
  actions: string[];
  selectedEntities: string[];
  selectedActions: {[key: string]: string};
  onEntitySelect: (entity: string) => void;
  onActionSelect: (entity: string, action: string) => void;
  entityCategories?: PiiEntityCategory[];
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
  onActionSelect,
  entityCategories = []
}) => {
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  
  // Create a lookup map to quickly find an entity's category
  const entityToCategoryMap = new Map<string, string>();
  entityCategories.forEach(category => {
    category.entities.forEach(entity => {
      entityToCategoryMap.set(entity, category.category);
    });
  });
  
  // Filter entities based on selected categories
  const filteredEntities = entities.filter(entity => {
    return selectedCategories.length === 0 || 
           selectedCategories.includes(entityToCategoryMap.get(entity) || '');
  });
  
  // Select all entities with a specified action
  const handleSelectAll = (action: string) => {
    entities.forEach(entity => {
      if (!selectedEntities.includes(entity)) {
        onEntitySelect(entity);
      }
      onActionSelect(entity, action);
    });
  };
  
  // Unselect all entities
  const handleUnselectAll = () => {
    selectedEntities.forEach(entity => {
      onEntitySelect(entity);
    });
  };

  // Format entity name for display
  const formatEntityName = (name: string) => {
    return name.replace(/_/g, ' ');
  };

  // Get action icon
  const getActionIcon = (action: string) => {
    switch(action) {
      case 'MASK':
        return <EyeInvisibleOutlined style={{ marginRight: 4 }} />;
      case 'BLOCK':
        return <StopOutlined style={{ marginRight: 4 }} />;
      default:
        return null;
    }
  };
  
  return (
    <div className="pii-configuration">
      <div className="flex justify-between items-center mb-5">
        <div className="flex items-center">
          <Title level={4} className="mb-0 font-semibold text-gray-800">Configure PII Protection</Title>
        </div>
        <Badge 
          count={selectedEntities.length} 
          showZero 
          style={{ backgroundColor: selectedEntities.length > 0 ? '#4f46e5' : '#d9d9d9' }}
          overflowCount={999}
        >
          <Text className="text-gray-500">{selectedEntities.length} items selected</Text>
        </Badge>
      </div>
      
      <div className="mb-6">
        <div className="flex items-center mb-2">
          <FilterOutlined className="text-gray-500 mr-1" />
          <Text className="text-gray-500 font-medium">Filter by category</Text>
        </div>
        <Select
          mode="multiple"
          placeholder="Select categories to filter by"
          style={{ width: '100%' }}
          onChange={setSelectedCategories}
          value={selectedCategories}
          allowClear
          showSearch
          optionFilterProp="children"
          className="mb-4"
          tagRender={(props) => (
            <Tag 
              color="blue" 
              closable={props.closable}
              onClose={props.onClose}
              className="mr-2 mb-2"
            >
              {props.label}
            </Tag>
          )}
        >
          {entityCategories.map(cat => (
            <Option key={cat.category} value={cat.category}>
              {cat.category}
            </Option>
          ))}
        </Select>
        
        {/* Quick Actions */}
        <div className="bg-gray-50 p-5 rounded-lg mb-6 border border-gray-200 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Text strong className="text-gray-700 text-base">Quick Actions</Text>
              <Tooltip title="Apply action to all PII types at once">
                <div className="ml-2 text-gray-400 cursor-help text-xs">ⓘ</div>
              </Tooltip>
            </div>
            <Button 
              type="default" 
              onClick={handleUnselectAll}
              disabled={selectedEntities.length === 0}
              icon={<CloseOutlined />}
              className="border-gray-300 hover:text-red-600 hover:border-red-300"
            >
              Unselect All
            </Button>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Button 
              type="default" 
              onClick={() => handleSelectAll('MASK')}
              className="flex items-center justify-center h-10 border-blue-200 hover:border-blue-300 hover:text-blue-700 bg-blue-50 hover:bg-blue-100 text-blue-600"
              block
              icon={<EyeInvisibleOutlined />}
            >
              Select All & Mask
            </Button>
            <Button 
              type="default" 
              onClick={() => handleSelectAll('BLOCK')}
              className="flex items-center justify-center h-10 border-red-200 hover:border-red-300 hover:text-red-700 bg-red-50 hover:bg-red-100 text-red-600"
              block
              icon={<StopOutlined />}
            >
              Select All & Block
            </Button>
          </div>
        </div>
      </div>
      
      {/* PII Type table */}
      <div className="border rounded-lg overflow-hidden shadow-sm">
        <div className="bg-gray-50 px-5 py-3 border-b flex">
          <Text strong className="flex-1 text-gray-700">PII Type</Text>
          <Text strong className="w-32 text-right text-gray-700">Action</Text>
        </div>
        <div className="max-h-[400px] overflow-y-auto">
          {filteredEntities.length === 0 ? (
            <div className="py-10 text-center text-gray-500">
              No PII types match your filter criteria
            </div>
          ) : (
            filteredEntities.map(entity => (
              <div 
                key={entity}
                className={`px-5 py-3 flex items-center justify-between hover:bg-gray-50 border-b ${
                  selectedEntities.includes(entity) ? 'bg-blue-50' : ''
                }`}
              >
                <div className="flex items-center flex-1">
                  <Checkbox 
                    checked={selectedEntities.includes(entity)} 
                    onChange={() => onEntitySelect(entity)}
                    className="mr-3"
                  />
                  <Text className={selectedEntities.includes(entity) ? 'font-medium text-gray-900' : 'text-gray-700'}>
                    {formatEntityName(entity)}
                  </Text>
                  {entityToCategoryMap.get(entity) && (
                    <Tag className="ml-2 text-xs" color="blue">{entityToCategoryMap.get(entity)}</Tag>
                  )}
                </div>
                <div className="w-32">
                  <Select
                    value={selectedEntities.includes(entity) ? (selectedActions[entity] || 'MASK') : 'MASK'}
                    onChange={(value) => onActionSelect(entity, value)}
                    style={{ width: 120 }}
                    disabled={!selectedEntities.includes(entity)}
                    className={`${!selectedEntities.includes(entity) ? 'opacity-50' : ''}`}
                    dropdownMatchSelectWidth={false}
                  >
                    {actions.map(action => (
                      <Option key={action} value={action}>
                        <div className="flex items-center">
                          {getActionIcon(action)}
                          {action}
                        </div>
                      </Option>
                    ))}
                  </Select>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default PiiConfiguration; 