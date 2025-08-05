import React, { useEffect, useState, useRef } from "react";
import { Form, Table, Input } from "antd";
import { Text, TextInput } from "@tremor/react";
import { Row, Col } from "antd";
import { QuestionCircleOutlined } from "@ant-design/icons";

const ConditionalPublicModelName: React.FC = () => {
  // Access the form instance
  const form = Form.useFormInstance();
  const [tableKey, setTableKey] = useState(0); // Add a key to force table re-render
  const [showPublicTooltip, setShowPublicTooltip] = useState(false);
  const [showLiteLLMTooltip, setShowLiteLLMTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState<'top' | 'bottom'>('top');
  const [liteLLMTooltipPosition, setLiteLLMTooltipPosition] = useState<'top' | 'bottom'>('top');
  
  const publicTooltipRef = useRef<HTMLDivElement>(null);
  const liteLLMTooltipRef = useRef<HTMLDivElement>(null);

  // Function to check if tooltip would fit above
  const checkTooltipPosition = (ref: React.RefObject<HTMLDivElement>, setPosition: (pos: 'top' | 'bottom') => void) => {
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect();
      const tooltipHeight = 300; // Approximate height of the tooltip
      const spaceAbove = rect.top;
      const spaceBelow = window.innerHeight - rect.bottom;
      
      if (spaceAbove < tooltipHeight && spaceBelow > tooltipHeight) {
        setPosition('bottom');
      } else {
        setPosition('top');
      }
    }
  };

  // Watch the 'model' field for changes and ensure it's always an array
  const modelValue = Form.useWatch('model', form) || [];
  const selectedModels = Array.isArray(modelValue) ? modelValue : [modelValue];
  const customModelName = Form.useWatch('custom_model_name', form);
  const showPublicModelName = !selectedModels.includes('all-wildcard');


  // Force table to re-render when custom model name changes
  useEffect(() => {
    if (customModelName && selectedModels.includes('custom')) {
      const currentMappings = form.getFieldValue('model_mappings') || [];
      const updatedMappings = currentMappings.map((mapping: any) => {
        if (mapping.public_name === 'custom' || mapping.litellm_model === 'custom') {
          return {
            public_name: customModelName,
            litellm_model: customModelName
          };
        }
        return mapping;
      });
      form.setFieldValue('model_mappings', updatedMappings);
      setTableKey(prev => prev + 1); // Force table re-render
    }
  }, [customModelName, selectedModels, form]);

  // Initial setup of model mappings when models are selected
  useEffect(() => {
    if (selectedModels.length > 0 && !selectedModels.includes('all-wildcard')) {
      // Check if we already have mappings that match the selected models
      const currentMappings = form.getFieldValue('model_mappings') || [];
      
      // Only update if the mappings don't exist or don't match the selected models
      const shouldUpdateMappings = currentMappings.length !== selectedModels.length ||
        !selectedModels.every(model =>
          currentMappings.some((mapping: { public_name: string; litellm_model: string }) => {
            if (model === 'custom') {
              return mapping.litellm_model === 'custom' || mapping.litellm_model === customModelName;
            }
            return mapping.litellm_model === model;
          }));
      
      if (shouldUpdateMappings) {
        const mappings = selectedModels.map((model: string) => {
          if (model === 'custom' && customModelName) {
            return {
              public_name: customModelName,
              litellm_model: customModelName
            };
          }
          return {
            public_name: model,
            litellm_model: model
          };
        });
        
        form.setFieldValue('model_mappings', mappings);
        setTableKey(prev => prev + 1); // Force table re-render
      }
    }
  }, [selectedModels, customModelName, form]);

  if (!showPublicModelName) return null;

  const columns = [
    {
      title: (
        <span style={{ display: 'flex', alignItems: 'center' }}>
          Public Model Name
          <div style={{ position: 'relative', display: 'inline-block' }} ref={publicTooltipRef}>
            <QuestionCircleOutlined 
              style={{ marginLeft: '4px', color: '#8c8c8c', cursor: 'help' }}
              onMouseEnter={() => {
                checkTooltipPosition(publicTooltipRef, setTooltipPosition);
                setShowPublicTooltip(true);
              }}
              onMouseLeave={() => setShowPublicTooltip(false)}
            />
            {showPublicTooltip && (
              <div style={{
                position: 'absolute',
                [tooltipPosition === 'top' ? 'bottom' : 'top']: '100%',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '500px',
                backgroundColor: 'rgba(0, 0, 0, 0.9)',
                color: '#ffffff',
                padding: '12px',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: 'normal',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"',
                zIndex: 1000,
                marginBottom: tooltipPosition === 'top' ? '8px' : '0',
                marginTop: tooltipPosition === 'bottom' ? '8px' : '0',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                border: '1px solid #333'
              }}>
                <div style={{ marginBottom: '8px', fontWeight: 'normal' }}>
                  The name you specify in your API calls to LiteLLM Proxy
                </div>
                <div style={{ marginBottom: '8px', fontWeight: 'normal' }}>
                  <strong>Example:</strong> If you choose <code style={{ backgroundColor: 'rgba(51, 51, 51, 0.8)', padding: '2px 4px', borderRadius: '2px' }}>openai/qwen-plus-latest</code> as the LiteLLM model, and name your public model <code style={{ backgroundColor: 'rgba(51, 51, 51, 0.8)', padding: '2px 4px', borderRadius: '2px' }}>example-name</code>
                </div>
                <div style={{ marginBottom: '8px', fontWeight: 'normal' }}>
                  <strong>Usage:</strong> You make an API call to the LiteLLM proxy with <code style={{ backgroundColor: 'rgba(51, 51, 51, 0.8)', padding: '2px 4px', borderRadius: '2px' }}>model = "example-name"</code>
                </div>
                <div style={{ fontWeight: 'normal' }}>
                  <strong>Result:</strong> LiteLLM sends <code style={{ backgroundColor: 'rgba(51, 51, 51, 0.8)', padding: '2px 4px', borderRadius: '2px' }}>qwen-plus-latest</code> to the provider
                </div>
                <div style={{
                  position: 'absolute',
                  [tooltipPosition === 'top' ? 'top' : 'bottom']: '100%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: 0,
                  height: 0,
                  borderLeft: '6px solid transparent',
                  borderRight: '6px solid transparent',
                  [tooltipPosition === 'top' ? 'borderTop' : 'borderBottom']: '6px solid rgba(0, 0, 0, 0.9)'
                }} />
              </div>
            )}
          </div>
        </span>
      ),
      dataIndex: 'public_name',
      key: 'public_name',
      render: (text: string, record: any, index: number) => {
        return (
          <TextInput
            value={text}
            onChange={(e) => {
              const newMappings = [...form.getFieldValue('model_mappings')];
              newMappings[index].public_name = e.target.value;
              form.setFieldValue('model_mappings', newMappings);
            }}
          />
        );
      }
    },
    {
      title: (
        <span style={{ display: 'flex', alignItems: 'center' }}>
          LiteLLM Model Name
          <div style={{ position: 'relative', display: 'inline-block' }} ref={liteLLMTooltipRef}>
            <QuestionCircleOutlined 
              style={{ marginLeft: '4px', color: '#8c8c8c', cursor: 'help' }}
              onMouseEnter={() => {
                checkTooltipPosition(liteLLMTooltipRef, setLiteLLMTooltipPosition);
                setShowLiteLLMTooltip(true);
              }}
              onMouseLeave={() => setShowLiteLLMTooltip(false)}
            />
            {showLiteLLMTooltip && (
              <div style={{
                position: 'absolute',
                [liteLLMTooltipPosition === 'top' ? 'bottom' : 'top']: '100%',
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: 'rgba(0, 0, 0, 0.9)',
                color: '#ffffff',
                padding: '8px 12px',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: 'normal',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"',
                zIndex: 1000,
                marginBottom: liteLLMTooltipPosition === 'top' ? '8px' : '0',
                marginTop: liteLLMTooltipPosition === 'bottom' ? '8px' : '0',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
                border: '1px solid #333',
                whiteSpace: 'nowrap'
              }}>
                The model name LiteLLM will send to the LLM API
                <div style={{
                  position: 'absolute',
                  [liteLLMTooltipPosition === 'top' ? 'top' : 'bottom']: '100%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: 0,
                  height: 0,
                  borderLeft: '6px solid transparent',
                  borderRight: '6px solid transparent',
                  [liteLLMTooltipPosition === 'top' ? 'borderTop' : 'borderBottom']: '6px solid rgba(0, 0, 0, 0.9)'
                }} />
              </div>
            )}
          </div>
        </span>
      ),
      dataIndex: 'litellm_model',
      key: 'litellm_model',
    }
  ];

  return (
    <>
      <Form.Item
        label="Model Mappings"
        name="model_mappings"
        tooltip="Map public model names to LiteLLM model names for load balancing"
        labelCol={{ span: 10 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        rules={[
          {
            required: true,
            validator: async (_, value) => {
              if (!value || value.length === 0) {
                throw new Error('At least one model mapping is required');
              }
              // Check if all mappings have valid public names
              const invalidMappings = value.filter((mapping: any) => 
                !mapping.public_name || mapping.public_name.trim() === ''
              );
              if (invalidMappings.length > 0) {
                throw new Error('All model mappings must have valid public names');
              }
            }
          }
        ]}
      >
        <Table 
          key={tableKey} // Add key to force re-render
          dataSource={form.getFieldValue('model_mappings')} 
          columns={columns} 
          pagination={false}
          size="small"
        />
      </Form.Item>
    </>
  );
};

export default ConditionalPublicModelName;