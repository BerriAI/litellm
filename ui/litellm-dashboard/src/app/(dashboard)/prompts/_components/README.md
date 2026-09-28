# Prompts Component

This component provides a view-only interface for viewing prompts in the LiteLLM dashboard, similar to the guardrails component.

## Components

### PromptsPanel (`prompts.tsx`)

- Main component that displays the prompts list
- Fetches prompts using the `getPromptsList` API call
- Handles loading states and error handling

### PromptTable (`prompt_table.tsx`)

- Table component that displays prompts data
- Uses Tanstack Table for sorting and filtering
- Shows: Prompt ID, Created At, Updated At, Type
- Supports clicking on prompt IDs to open detailed view

### PromptInfoView (`prompt_info.tsx`)

- Detail view component for individual prompts
- Shows comprehensive prompt information including metadata and parameters
- Three-tab interface: **Overview**, **Details** (admin-only), and **Raw JSON**
- **Overview**: Shows formatted prompt information with key details
- **Details**: Shows structured breakdown of all prompt data (admin users only)
- **Raw JSON**: Shows exactly what the API returns with copy-to-clipboard functionality
- Includes copy-to-clipboard functionality for prompt ID and raw JSON
- Similar structure to GuardrailInfoView component

## Usage

The component is integrated into the main application at:

- **Navigation**: Available in the left sidebar under "Experimental" > "Prompts" (admin role required)
- **Routing**: Accessible via `?page=prompts` URL parameter
- **API**: Uses `getPromptsList` and `getPromptInfo` functions from `networking.tsx`
- **Detail View**: Click any prompt ID to view detailed information

## Props

```typescript
interface PromptsProps {
  accessToken: string | null;
  userRole?: string;
}
```

## Data Structure

The component expects prompts with the following structure:

```typescript
interface PromptItem {
  prompt_id?: string;
  prompt_name: string | null;
  prompt_info: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}
```

## Integration

The component is fully integrated into the main application:

1. **Left Navigation**: Added to `leftnav.tsx` with `FileTextOutlined` icon
2. **Main Routing**: Added to `page.tsx` with proper routing logic
3. **Permissions**: Restricted to admin roles (same as guardrails)

## Future Enhancements

- ✅ Add prompt detail view (similar to GuardrailInfoView) - **COMPLETED**
- Add create/edit/delete functionality
- Add bulk operations
- Add search and filtering capabilities
- Add export functionality
