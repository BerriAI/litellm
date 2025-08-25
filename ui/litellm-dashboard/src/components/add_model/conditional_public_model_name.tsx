import React, { useEffect, useState, useMemo } from "react";
import { Form, Table, Tooltip, Empty, Typography } from "antd";
import { TextInput } from "@tremor/react";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Providers } from "../provider_info_helpers";

const { Link } = Typography;

interface ConditionalPublicModelNameProps {
  providerModels: string[];
  showTitle?: boolean;
}

const ConditionalPublicModelName: React.FC<ConditionalPublicModelNameProps> = ({ providerModels, showTitle = false }) => {
  const form = Form.useFormInstance();


  // Watch the 'model' field for changes and ensure it's always an array
  const modelValue = Form.useWatch('model', form);
  const selectedModels = useMemo(() => {
    const value = modelValue || [];
    return Array.isArray(value) ? value : [value];
  }, [modelValue]);
  const customModelName = Form.useWatch('custom_model_name', form);
  const selectedProvider = Form.useWatch('custom_llm_provider', form);
  const isWildcardSelected = selectedModels.includes('all-wildcard');
  
  const modelMappings = Form.useWatch('model_mappings', form) || [];
  const hasSelectedModels = selectedModels.length > 0;
  
  // Force table to re-render when custom model name changes
  useEffect(() => {
    if (customModelName && selectedModels.includes('custom')) {
      const currentMappings = form.getFieldValue('model_mappings') || [];
      const updatedMappings = currentMappings.map((mapping: any) => {
        if (mapping.public_name === 'custom' || mapping.litellm_model === 'custom') {
          if (selectedProvider === Providers.Azure) {
            return {
              public_name: customModelName,
              litellm_model: `azure/${customModelName}`
            };
          }
          return {
            public_name: customModelName,
            litellm_model: customModelName
          };
        }
        return mapping;
      });
      form.setFieldValue('model_mappings', updatedMappings);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customModelName, selectedModels, selectedProvider]);

  // Initial setup of model mappings when models are selected
  useEffect(() => {
    if (selectedModels.length === 0) {
      // Clear mappings when no models are selected
      form.setFieldValue('model_mappings', []);
    } else if (selectedModels.includes('all-wildcard')) {
      // When wildcard is selected, show all provider models
      if (providerModels.length > 0 && selectedProvider) {
        const wildcardMappings = providerModels.map((model: string) => {
          if (selectedProvider === Providers.Azure) {
            return {
              public_name: '', // Empty initially, user can customize
              litellm_model: `azure/${model}`
            };
          }
          return {
            public_name: '', // Empty initially, user can customize
            litellm_model: model
          };
        });
        form.setFieldValue('model_mappings', wildcardMappings);
      }
    } else if (selectedModels.length > 0) {
      // Check if we already have mappings that match the selected models
      const currentMappings = form.getFieldValue('model_mappings') || [];
      
      // Only update if the mappings don't exist or don't match the selected models
      const shouldUpdateMappings = currentMappings.length !== selectedModels.length ||
        !selectedModels.every(model =>
          currentMappings.some((mapping: { public_name: string; litellm_model: string }) => {
            if (model === 'custom') {
              return mapping.litellm_model === 'custom' || mapping.litellm_model === customModelName;
            }
            if (selectedProvider === Providers.Azure) {
              return mapping.litellm_model === `azure/${model}`;
            }
            return mapping.litellm_model === model;
          }));
      
      if (shouldUpdateMappings) {
        const mappings = selectedModels.map((model: string) => {
          if (model === 'custom' && customModelName) {
            if (selectedProvider === Providers.Azure) {
              return {
                public_name: '', // Empty initially, user can customize
                litellm_model: `azure/${customModelName}`
              };
            }
            return {
              public_name: '', // Empty initially, user can customize
              litellm_model: customModelName
            };
          }
          if (selectedProvider === Providers.Azure) {
            return {
              public_name: '', // Empty initially, user can customize
              litellm_model: `azure/${model}`
            };
          }
          return {
            public_name: '', // Empty initially, user can customize
            litellm_model: model
          };
        });
        
        form.setFieldValue('model_mappings', mappings);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModels, customModelName, selectedProvider, providerModels]);

  // Always show the table structure, but handle empty state within the table

  const publicNameTooltipContent = (
    <>
      <div className="mb-2 font-normal">
        <strong>Optional:</strong> Custom name for your API calls to LiteLLM Proxy. If left empty, the LiteLLM Model Name will be used as the public name.
      </div>
        <div className="mb-2 font-normal">
          <strong>Example:</strong> If you name your public model <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">example-name</code> 
           , and choose <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">openai/qwen-plus-latest</code> as the LiteLLM model 
      </div>
      <div className="mb-2 font-normal">
        <strong>Usage:</strong> You make an API call to the LiteLLM proxy with <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">model = &quot;example-name&quot;</code>
      </div>
      <div className="font-normal">
        <strong>Result:</strong> LiteLLM sends <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">qwen-plus-latest</code> to the provider
      </div>
    </>
  );

  const liteLLMModelTooltipContent = (
    <div>The model name LiteLLM will send to the LLM API</div>
  );

  const columns = [
    {
      title: (
        <div className="flex items-center justify-center w-full">
          <span className="flex items-center justify-center">
            Public Model (Custom Name)
            <Tooltip title={publicNameTooltipContent}>
              <InfoCircleOutlined className="ml-1 text-gray-400" />
            </Tooltip>
          </span>
        </div>
      ),
      dataIndex: 'public_name',
      key: 'public_name',
      width: '50%',
      render: (text: string, record: any, index: number) => {
        // If public_name is empty, use litellm_model as the effective value but show placeholder
        const displayValue = text || '';
        const effectiveValue = text || record.litellm_model;
        
        // For special providers (Azure, etc.), show generic placeholder
        // For standard providers, show the actual LiteLLM model name as placeholder
        const isSpecialProvider = selectedProvider === Providers.Azure || 
                                 selectedProvider === Providers.OpenAI_Compatible || 
                                 selectedProvider === Providers.Ollama;
        
        const placeholderText = isSpecialProvider 
          ? "Optional custom name" 
          : record.litellm_model;
        
        return (
          <TextInput
            value={displayValue}
            placeholder={placeholderText}
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
        <div className="flex items-center justify-center w-full">
          <span className="flex items-center justify-center">
            LiteLLM Model Name
            <Tooltip title={liteLLMModelTooltipContent}>
              <InfoCircleOutlined className="ml-1 text-gray-400" />
            </Tooltip>
          </span>
        </div>
      ),
      dataIndex: 'litellm_model',
      key: 'litellm_model',
      width: '50%',
      render: (text: string, record: any, index: number) => {
        // LiteLLM Model Name is read-only and displays the actual model name
        return (
          <div className="text-gray-700 font-medium">
            {text}
          </div>
        );
      }
    }
  ];

  return (
    <>
      {/* Title above the table */}
      {showTitle && (
        <div className="mb-4">
          <div className="flex items-center gap-2">
            <h4 className="text-lg font-semibold text-gray-900 mb-0">Model Mappings</h4>
            <Tooltip title="Map public model names to LiteLLM model names for load balancing">
              <InfoCircleOutlined className="text-gray-400" />
            </Tooltip>
          </div>
          <p className="text-sm text-gray-600 mt-1">
            Configure how your models will be mapped for{" "}
            <Link
              href="https://docs.litellm.ai/docs/proxy/load_balancing"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800"
            >
              load balancing
            </Link>
          </p>
        </div>
      )}

             {/* Centered table */}
       <div className="flex justify-center mb-4">
         <div className="w-full max-w-4xl">
           <Form.Item
             name="model_mappings"
             labelCol={{ span: 0 }}
             wrapperCol={{ span: 24 }}
                           rules={[
                {
                  required: true,
                  validator: async (_, value) => {
                    if (!value || value.length === 0) {
                      throw new Error('At least one model mapping is required');
                    }
                    // Check if all mappings have valid litellm_model names (public_name can be empty)
                    const invalidMappings = value.filter((mapping: any) => 
                      !mapping.litellm_model || mapping.litellm_model.trim() === ''
                    );
                    if (invalidMappings.length > 0) {
                      throw new Error('All model mappings must have valid LiteLLM model names');
                    }
                  }
                }
              ]}
           >
             <Table 
               dataSource={modelMappings} 
               columns={columns} 
               pagination={false}
               size="small"
               tableLayout="fixed"
               className="model-mappings-table border border-gray-200 rounded-lg overflow-hidden [&_.ant-table-thead>tr>th]:bg-gray-200"
               rowClassName={(record, index) => index % 2 === 0 ? '' : 'bg-gray-100'}
               locale={{
                 emptyText: hasSelectedModels ? (
                   <Empty 
                     image={Empty.PRESENTED_IMAGE_SIMPLE}
                     description="No model mappings available"
                   />
                 ) : (
                   <Empty 
                     image={Empty.PRESENTED_IMAGE_SIMPLE}
                     description={
                       !selectedProvider 
                         ? "Select a provider and model(s) to configure mappings"
                         : "Select models to configure mappings"
                     }
                   />
                 )
               }}
             />
           </Form.Item>
         </div>
       </div>
    </>
  );
};

export default ConditionalPublicModelName;