import React, { useRef, useEffect, useState } from "react";
import { Form, Select as AntSelect } from "antd";
import { TextInput } from "@tremor/react";
import { Providers } from "../provider_info_helpers";

interface LiteLLMModelNameFieldProps {
  selectedProvider: Providers | null;
  providerModels: string[];
  getPlaceholder: (provider: Providers) => string;
}

const LiteLLMModelNameField: React.FC<LiteLLMModelNameFieldProps> = ({
  selectedProvider,
  providerModels, 
  getPlaceholder,
}) => {
  const form = Form.useFormInstance();
  const previousValuesRef = useRef<string[]>([]);
  const selectRef = useRef<any>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  // Initialize and update the previous values ref when form changes
  useEffect(() => {
    const currentModel = form.getFieldValue('model') || [];
    const currentValues = Array.isArray(currentModel) ? currentModel : [currentModel];
    previousValuesRef.current = currentValues;
  }, [form]);

  const handleModelChange = (value: string | string[]) => {
    // Ensure value is always treated as an array
    const values = Array.isArray(value) ? value : [value];
    const previousValues = previousValuesRef.current;
    
    // Implement exclusive selection logic for "all-wildcard"
    let finalValues = values;
    
    const hasAllInValues = values.includes("all-wildcard");
    const hasAllInPrevious = previousValues.includes("all-wildcard");
    const hasNonAllInValues = values.some(v => v !== "all-wildcard");
    
    // Debug logging to understand the issue (remove in production)
    console.log("Model selection debug:", {
      previousValues,
      incomingValues: values,
      hasAllInValues,
      hasAllInPrevious,
      hasNonAllInValues
    });
    
    if (hasAllInValues && hasNonAllInValues) {
      // If the new selection contains both ALL and other models, determine which was just added
      if (hasAllInPrevious) {
        // ALL was already selected, user added an individual model -> keep only the individual models
        finalValues = values.filter(v => v !== "all-wildcard");
      } else {
        // Individual models were selected, user added ALL -> keep only ALL
        finalValues = ["all-wildcard"];
      }
      
      // Immediately update the form field when exclusivity rules apply
      console.log("Applying exclusivity rule, finalValues:", finalValues);
      
      // Use setTimeout to ensure the Select component updates properly
      setTimeout(() => {
        form.setFieldsValue({ model: finalValues });
        
        // Only close dropdown when switching FROM individual models TO ALL
        // (not when switching from ALL to individual models)
        if (!hasAllInPrevious && finalValues.includes("all-wildcard")) {
          // User had individual models and selected ALL -> close dropdown
          setDropdownOpen(false);
          if (selectRef.current) {
            selectRef.current.blur();
          }
        }
        // When switching from ALL to individual models, keep dropdown open for more selections
      }, 0);
      
      // Update the previous values ref and return early to prevent double processing
      previousValuesRef.current = finalValues;
      return;
    } else if (hasAllInValues && !hasNonAllInValues) {
      // Only ALL is selected -> keep ALL
      finalValues = ["all-wildcard"];
    } else {
      // Only individual models are selected (or empty) -> keep them
      finalValues = values;
    }
    
    // Update the previous values ref for next time
    previousValuesRef.current = finalValues;
    
    // If "all-wildcard" is in final selection, clear the model_name field
    if (finalValues.includes("all-wildcard")) {
      form.setFieldsValue({ model_name: undefined });
    } else if (finalValues.length === 0 || (finalValues.length === 1 && finalValues[0] === '')) {
      // If selection is cleared (empty array or single empty string), clear the mappings
      form.setFieldsValue({ 
        model: [],
        model_mappings: []
      });
    } else {
      // Only update if the value has actually changed
      if (JSON.stringify(previousValues) !== JSON.stringify(finalValues)) {

        // Create mappings first
        const mappings = finalValues.map(model => {
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
        
        // Update both fields in one call to reduce re-renders
        form.setFieldsValue({ 
          model: finalValues,
          model_mappings: mappings
        });
        
      }
    }
  };

  const handleAzureDeploymentNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const deploymentName = e.target.value;

    // Create mapping with Azure-specific format
    const mappings = deploymentName ? [{
      public_name: '', // Empty initially, user can customize
      litellm_model: `azure/${deploymentName}`
    }] : [];
    
    // Update both fields
    form.setFieldsValue({ 
      model: deploymentName || [],
      model_mappings: mappings
    });
  };

  const handleGenericDeploymentNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const deploymentName = e.target.value;

    // Create mapping with generic format (for OpenAI_Compatible and Ollama)
    const mappings = deploymentName ? [{
      public_name: '', // Empty initially, user can customize
      litellm_model: deploymentName
    }] : [];
    
    // Update both fields
    form.setFieldsValue({ 
      model: deploymentName || [],
      model_mappings: mappings
    });
  };

  // Handle custom model name changes
  const handleCustomModelNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const customName = e.target.value;

    // Immediately update the model mappings
    const currentMappings = form.getFieldValue('model_mappings') || [];
    const updatedMappings = currentMappings.map((mapping: any) => {
      if (mapping.public_name === 'custom' || mapping.litellm_model === 'custom') {
        if (selectedProvider === Providers.Azure) {
          return {
            public_name: '', // Empty initially, user can customize
            litellm_model: `azure/${customName}`
          };
        }
        return {
          public_name: '', // Empty initially, user can customize
          litellm_model: customName
        };
      }
      return mapping;
    });
    
    form.setFieldsValue({ model_mappings: updatedMappings });
  };

  return (
    <>
      <Form.Item
        label="LiteLLM Model(s)"
        name="model"
        tooltip={selectedProvider === Providers.Azure 
          ? "Enter your Azure deployment name (e.g., 'my-gpt4-deployment')"
          : selectedProvider === Providers.OpenAI_Compatible
          ? "Enter the model name your endpoint expects (e.g., 'gpt-3.5-turbo')"
          : selectedProvider === Providers.Ollama
          ? "Enter your Ollama model name (e.g., 'llama2', 'codellama:13b')"
          : "Select model(s) to use."
        }
        rules={[{ required: true, message: `Please enter ${selectedProvider === Providers.Azure ? 'a deployment name' : 'at least one model'}.` }]}
        labelCol={{ span: 24 }}
        labelAlign="left"
      >
        {(selectedProvider === Providers.Azure) || 
         (selectedProvider === Providers.OpenAI_Compatible) || 
         (selectedProvider === Providers.Ollama) ? (
          <TextInput 
            placeholder={selectedProvider ? getPlaceholder(selectedProvider) : "Select a provider first"}
            disabled={!selectedProvider}
            onChange={selectedProvider === Providers.Azure ? handleAzureDeploymentNameChange : handleGenericDeploymentNameChange}
          />
        ) : (
          <AntSelect
            ref={selectRef}
            mode="multiple"
            allowClear
            showSearch
            open={dropdownOpen}
            onDropdownVisibleChange={setDropdownOpen}
            placeholder={selectedProvider ? "Select models" : "Select a provider first"}
            onChange={handleModelChange}
            optionFilterProp="children"
            filterOption={(input, option) =>
              (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
            }
            options={selectedProvider && providerModels.length > 0 ? [
              {
                label: 'Custom Model Name (Enter below)',
                value: 'custom'
              },
              {
                label: `All ${selectedProvider} Models (Wildcard)`,
                value: 'all-wildcard'
              },
              ...providerModels.map(model => ({
                label: model,
                value: model
              }))
            ] : selectedProvider ? [
              {
                label: 'Custom Model Name (Enter below)',
                value: 'custom'
              },
              {
                label: `All ${selectedProvider} Models (Wildcard)`,
                value: 'all-wildcard'
              }
            ] : []}
            style={{ width: '100%' }}
            disabled={!selectedProvider}
          />
        )}
      </Form.Item>

      {/* Custom Model Name field */}
      <Form.Item
        noStyle
        shouldUpdate={(prevValues, currentValues) => 
          prevValues.model !== currentValues.model
        }
      >
        {({ getFieldValue }) => {
          const selectedModels = getFieldValue('model') || [];
          const modelArray = Array.isArray(selectedModels) ? selectedModels : [selectedModels];
          return modelArray.includes('custom') && (
            <Form.Item
              name="custom_model_name"
              rules={[{ required: true, message: "Please enter a LiteLLM model name." }]}
              className="mt-2"
              labelCol={{ span: 24 }}
              labelAlign="left"
            >
              <TextInput 
                placeholder={selectedProvider === Providers.Azure ? "Enter Azure deployment name" : "Enter LiteLLM model name"}
                onChange={handleCustomModelNameChange}
              />
            </Form.Item>
          );
        }}
              </Form.Item>
        
        {/* Removed redundant explanatory text - having both tooltip and help text below the field 
            was overkill and could be more confusing than clarifying for users */}
      </>
  );
};

export default LiteLLMModelNameField;