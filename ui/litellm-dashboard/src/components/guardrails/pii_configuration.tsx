import React, { useState } from 'react';
import { Typography, Select, Button, Checkbox, Input } from 'antd';
import { TextInput } from '@tremor/react';

const { Title, Text } = Typography;
const { Option } = Select;
const { Search } = Input;

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
  const [searchText, setSearchText] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  // Create a lookup map to quickly find an entity's category
  const entityToCategoryMap = new Map<string, string>();
  entityCategories.forEach(category => {
    category.entities.forEach(entity => {
      entityToCategoryMap.set(entity, category.category);
    });
  });
  
  // Filter entities based on search text and selected category
  const filteredEntities = entities.filter(entity => {
    const matchesSearch = searchText === '' || entity.toLowerCase().includes(searchText.toLowerCase());
    const matchesCategory = !selectedCategory || entityToCategoryMap.get(entity) === selectedCategory;
    return matchesSearch && matchesCategory;
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
  
  return (
    <div className="pii-configuration">
      <Title level={4} className="mb-4">Configure PII Protection</Title>
      
      <div className="mb-4">
        <Search
          placeholder="Search PII types..."
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          style={{ width: '100%' }}
          className="mb-4"
        />
        
        {/* Quick Actions */}
        <div className="bg-gray-50 p-4 rounded-lg mb-6">
          <Title level={5} className="mb-3">Quick Actions</Title>
          <div className="grid grid-cols-2 gap-4">
            <Button 
              type="default" 
              onClick={() => handleSelectAll('MASK')}
              block
            >
              Select All & Mask
            </Button>
            <Button 
              type="default" 
              onClick={() => handleSelectAll('BLOCK')}
              block
            >
              Select All & Block
            </Button>
          </div>
        </div>
      </div>
      
      {/* Tabs */}
      <div className="mb-4">
        <div className="flex border-b mb-4">
          <div className="py-2 px-4 border-b-2 border-blue-500 text-blue-500 font-medium">
            Categories
          </div>
          <div className="py-2 px-4 text-gray-500 font-medium">
            All PII Types
          </div>
        </div>
        
        {/* Category filters */}
        <div className="flex flex-wrap gap-2 mb-4">
          <Button
            type={selectedCategory === null ? 'primary' : 'default'}
            onClick={() => setSelectedCategory(null)}
            className="mr-2"
          >
            All
          </Button>
          {entityCategories.map(cat => (
            <Button
              key={cat.category}
              type={selectedCategory === cat.category ? 'primary' : 'default'}
              onClick={() => setSelectedCategory(cat.category)}
              className="mr-2"
            >
              {cat.category}
            </Button>
          ))}
        </div>
        
        {/* PII Type table */}
        <div className="border rounded-lg overflow-hidden">
          <div className="bg-gray-50 px-4 py-3 border-b flex">
            <Text strong className="flex-1">PII Type</Text>
            <Text strong className="w-32 text-right">Action</Text>
          </div>
          <div className="max-h-[400px] overflow-y-auto">
            {filteredEntities.map(entity => (
              <div 
                key={entity}
                className={`px-4 py-3 flex items-center justify-between hover:bg-gray-50 border-b ${selectedEntities.includes(entity) ? 'bg-blue-50' : ''}`}
              >
                <div className="flex items-center flex-1">
                  <Checkbox 
                    checked={selectedEntities.includes(entity)} 
                    onChange={() => onEntitySelect(entity)}
                    className="mr-3"
                  />
                  <Text>{entity.replace(/_/g, ' ')}</Text>
                </div>
                <div className="w-32">
                  <Select
                    value={selectedEntities.includes(entity) ? (selectedActions[entity] || 'BLOCK') : 'BLOCK'}
                    onChange={(value) => onActionSelect(entity, value)}
                    style={{ width: 120 }}
                    disabled={!selectedEntities.includes(entity)}
                  >
                    {actions.map(action => (
                      <Option key={action} value={action}>{action}</Option>
                    ))}
                  </Select>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PiiConfiguration; 