import React, { useState, useEffect } from 'react';
import { Typography, Select, Tag, Divider, Input, Modal } from 'antd';

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
  const [showPiiTypeModal, setShowPiiTypeModal] = useState(false);
  
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
  
  // Open the modal when the user clicks the PII type input
  const handlePiiTypeClick = () => {
    setShowPiiTypeModal(true);
  };
  
  // Close the modal
  const handleModalClose = () => {
    setShowPiiTypeModal(false);
  };
  
  return (
    <>
      <Divider>PII Configuration</Divider>
      
      {/* PII Type Selection */}
      <div className="mb-4">
        <div className="mb-2">
          <Input 
            placeholder="Choose PII type"
            value={selectedEntities.length > 0 ? `${selectedEntities.length} PII type(s) selected` : 'Choose PII type'}
            readOnly
            onClick={handlePiiTypeClick}
            suffix={<i className="fas fa-chevron-down" />}
          />
        </div>
      </div>
      
      {/* PII Type Selection Modal */}
      <Modal
        title="Choose PII type"
        open={showPiiTypeModal}
        onCancel={handleModalClose}
        footer={null}
        width={600}
      >
        <div className="mb-4">
          <Search
            placeholder="Choose PII type"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ width: '100%' }}
          />
        </div>
        
        {/* Category list */}
        <div className="mb-4">
          <Title level={5} className="mb-2">Categories</Title>
          <div className="flex flex-wrap gap-2">
            <Tag 
              className="cursor-pointer text-sm py-1 px-3"
              color={selectedCategory === null ? 'blue' : 'default'}
              onClick={() => setSelectedCategory(null)}
            >
              All
            </Tag>
            {entityCategories.map(cat => (
              <Tag 
                key={cat.category}
                className="cursor-pointer text-sm py-1 px-3"
                color={selectedCategory === cat.category ? 'blue' : 'default'}
                onClick={() => setSelectedCategory(cat.category)}
              >
                {cat.category}
              </Tag>
            ))}
          </div>
        </div>
        
        {/* Entity list */}
        <div>
          {entityCategories.filter(cat => 
            selectedCategory === null || selectedCategory === cat.category
          ).map(category => (
            <div key={category.category} className="mb-4">
              <Title level={5} className="ml-2 mb-2">{category.category}</Title>
              <div className="flex flex-col gap-2">
                {category.entities
                  .filter(entity => filteredEntities.includes(entity))
                  .map(entity => (
                    <div 
                      key={entity}
                      className={`px-4 py-2 flex items-center justify-between cursor-pointer ${selectedEntities.includes(entity) ? 'bg-gray-100' : ''}`}
                      onClick={() => onEntitySelect(entity)}
                    >
                      <Text>{entity.replace(/_/g, ' ')}</Text>
                      {selectedEntities.includes(entity) && (
                        <span className="text-blue-500">
                          <i className="fas fa-check" />
                        </span>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      </Modal>
      
      {/* Action configuration for selected entities */}
      {selectedEntities.length > 0 && (
        <div className="mb-4">
          <Title level={5}>Configure Actions</Title>
          {selectedEntities.map(entity => (
            <div key={entity} className="flex items-center justify-between mb-2">
              <Text>{entity.replace(/_/g, ' ')}</Text>
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