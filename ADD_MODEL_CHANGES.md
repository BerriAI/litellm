# Add Model Interface Changes

This document details all changes made as part of the **LiteLLM Takehome: Redesign Model Add** assignment. The changes improve UX, fix critical issues, and enhance the overall model configuration experience. 

NOTE: These are mainly UI fixes, and backend compatibility isn't thoroughly tested (Especially for updated advanced settings)


### Specific Concerns Addressed Briefly (Detailed explanation below)

#### What is 'public model'?
Renamed this section, added more thorough tooltip, and reorganized layout to properly guide user and help them understand better

#### How does load balancing work?
Added link within grouped "Model Mappings" description to redirect users that may be unaware of the concept

#### What is the difference between public model and litellm model?
Renamed table headers slightly, altered empty table to prompt users through setup, introduced properly styled placeholder text (with appropriate default behavior) to let users know intuitively that Public Model field is editable

#### What is ‘Team-BYOK model’? 
Organized into descriptive group, Improved tooltip information, and linked to URL (placeholder) that provides a more ample description 

#### Advanced Users
None of these solutions are functional, they are simply prototypes
- Solution A displayed in Advanced Options, under Add Models tab. However, this tab doesn't seem like the appropriate place to display this, which is why there are two options.
- Solution B in All Models tab, simply added as an exrta field

# Model Mappings
## Reconfigured Layout
- **Issue**: It was not obvious that “Model Mappings” option was dependent on the previous two options (Provider & LiteLLM Models). 
- **Solution**: Layout was changed to ensure that Provider and LiteLLM Models visually appear to be required for filing in the table. Table text now (in its empty state) also prompts users to follow the desired flow of actions.

## Provider Option
- **Issue**: It was not obvious that "Model Mappings" option was dependent on the previous two options (Provider & LiteLLM Models)
- **Solution**: Layout was changed to ensure that Provider and LiteLLM Models visually appear to be required for filing in the table. Table text now (in its empty state) also prompts users to follow the desired flow of actions


## LiteLLM Model Name(s) Option

### 1. Fixed Input Type When No Provider Selected
- **Issue**: Field showed as text input when no provider was selected, allowing users to enter invalid data before selecting a provider
- **Solution**: Field now starts disabled until provider is selected with dependency checks and disabled state management to prevent invalid input and clarify field dependencies

### 2. Implemented Hybrid Input Approach
- **Issue**: All providers treated the same way causing inconsistent UX between deployment-based vs model-based providers that require different input patterns
- **Solution**: Different input types based on provider requirements (Azure, OpenAI_Compatible, Ollama use TextInput for custom names; other providers use dropdown with predefined models) providing appropriate input method for each provider's architecture

### 3. Enhanced Provider-Specific Tooltips
- **Issue**: Generic tooltip didn't provide specific guidance for different providers, leaving users unclear about expected input format
- **Solution**: Dynamic tooltips with provider-specific instructions (Azure: deployment name examples, OpenAI_Compatible: endpoint model names, Ollama: model name examples, Others: selection guidance) providing clear, contextual guidance for each provider type

### 4. Removed Redundant Description Text
- **Issue**: Static description "The model name LiteLLM will send to the LLM API" created visual clutter with information already covered by tooltips
- **Solution**: Removed static description and enhanced tooltips instead, creating cleaner interface with better contextual help

### 5. Implemented Exclusive ALL Selection Logic
- **Issue**: Users could select "ALL [provider] Models" alongside individual models, creating confusing and conflicting configurations
- **Solution**: Mutual exclusivity between ALL and individual selections with detection logic that replaces ALL with individual selections and vice versa, plus smart dropdown closing for clear, predictable selection behavior preventing conflicts


## Model Mappings Table
### 1. Fixed Table Visibility with Wildcard Selection
- **Issue**: Table disappeared when "ALL [provider] Models" was selected, preventing users from seeing or configuring available models with wildcard
- **Solution**: Show table with all provider models when wildcard is selected by modifying visibility logic and adding `providerModels` prop, allowing users to see and customize public names for all models

### 2. Eliminated Redundant Text Duplication
- **Issue**: "Public Model Name" and "LiteLLM Model Name" showed duplicate information with unclear distinction between editable and display fields
- **Solution**: Made LiteLLM Model Name read-only display and Public Model Name editable with dynamic placeholder text and backend defaults, providing clear separation with proper default behavior

### 3. Fixed Table Clearing Issue
- **Issue**: Table retained stale data when model selection was cleared, causing users to see outdated mappings after clearing selections
- **Solution**: Enhanced clearing logic across all input handlers with detection for empty selections, updated text input handlers, and enhanced `useEffect` to handle empty arrays for consistent table clearing when selections are removed

### 4. Enhanced Table Visual Design
- **Issue**: Table headers blended with section background with poor visual separation, making it difficult to distinguish headers and scan table content
- **Solution**: Professional styling with alternating colors, clear boundaries, rounded corners, header styling, and fixed column widths for professional appearance with excellent readability

### 5. Fixed Auto-Fill Issues
- **Issue**: Browser auto-filled credential fields with saved passwords/emails, causing unexpected values to appear in API Key and Existing Credentials fields
- **Solution**: Comprehensive auto-fill prevention using `autoComplete="new-password"` for TextInput fields and `autoComplete="off"` for Select components, preventing unwanted browser auto-fill while preserving intentional defaults

### 6. Improved Empty State Display
- **Issue**: Table completely disappeared when no models selected, causing users to see blank space instead of helpful guidance
- **Solution**: Always show table structure with contextual empty states using Ant Design `Empty` component with dynamic messages for different scenarios, providing clear guidance on required actions with appropriate visual feedback


# Authentication & Connection

## Mode Option

### 1. Converted Descriptions to Tooltips
- **Issue**: Standalone description text created visual clutter, making interface feel crowded with unnecessary text blocks
- **Solution**: Moved descriptions to tooltips (Mode Field health check description and Existing Credentials selection guidance) for cleaner, more compact interface with contextual help on hover

### 2. Added Mode Opt-Out Option
- **Issue**: Users couldn't clear Mode selection once chosen despite field being optional, providing no way to reset optional field after selection
- **Solution**: Added "None (Default)" option with `allowClear` prop and placeholder indicating optional nature, allowing users to easily opt out of optional selections

## Credentials Option

### 1. Reorganized Credentials into Clear Categories
- **Issue**: Credentials split with confusing OR divider that was present whether API key was used or if existing credentials was used
- **Solution**: Split credentials into new and existing categories with clear visual separation, eliminating the confusing OR divider and making the relationship between options obvious

### 2. Clarified API Key Requirements
- **Issue**: API Key being required was confusing to users who weren't sure when it was needed
- **Solution**: Made it visually clear that API Key is only required UNLESS you choose from an existing credential, providing clear conditional requirements based on credential selection

## Additional Model Info Settings

### 1. Grouped Team-BYOK Model and Model Access Group
- **Issue**: Team-BYOK Model and Model Access Group settings were scattered and lacked visual organization
- **Solution**: Grouped these related settings together with clear visual separation and logical organization for better user understanding of model access controls

### 2. Updated Team-BYOK Model Switch Icon
- **Issue**: Switch icon for Team-BYOK Model didn't match the existing implementation used in advanced settings
- **Solution**: Replaced switch icon with icon that matches existing implementation in advanced settings for consistent visual design across the interface

## Advanced Settings

### 1. Grouped Settings into Distinct Categories
- **Issue**: Advanced settings were disorganized and difficult to navigate, lacking clear categorization
- **Solution**: Grouped settings into distinct categories with clear visual separation and logical organization for improved navigation and user understanding

### 2. Fixed Custom Pricing Position Issue
- **Issue**: Custom pricing was opening below Guardrails option instead of in the correct position
- **Solution**: Fixed positioning logic to ensure Custom pricing appears in the appropriate location within the settings hierarchy

### 3. Implemented Weight Management for Load Balancing
- **Issue**: The LiteLLM admin UI didn't support setting weight parameters for load balancing. While weights could be configured in YAML files, there were limitations when managing models through the UI - no weight field in model creation, models with weight in litellm_params became uneditable, and users couldn't modify existing model weights through the UI
- **Solution**: 
  - **Added Weight Section**: Created a dedicated "Manage Model Weights" section in Advanced Settings with an interactive table showing model configurations
  - **Editable Weight Table**: Implemented a table with editable weight fields where users can directly modify individual model weights and see real-time updates to request distribution
  - **Real-time Calculations**: Added dynamic calculation of "Requests handled by Model" percentages that update immediately when weights are changed
  - **Dual Model Name Display**: Split model information into "Public Model (Custom Name)" and "LiteLLM Model Name" columns for clarity
  - **Form Integration**: Integrated weight field with existing form submission logic so weights are properly included in litellm_params when models are created
  - **Edit Model Support**: Added weight field to the edit model modal so existing model weights can be modified through the UI
  - **Table Display**: Added weight column to the main model list table showing current weight values with proper default handling
  - **Validation**: Implemented positive number validation for weight values with appropriate error handling




## User Experience Impact

### Before Changes
- Confusing field dependencies
- Inconsistent input methods across providers
- Poor visual hierarchy
- Browser auto-fill issues
- Conflicting selection states

### After Changes
- Clear dependency relationships
- Provider-appropriate input methods
- Professional visual design
- Stable, predictable behavior
- Comprehensive guidance system

