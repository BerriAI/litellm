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
- Shows: Prompt ID, Name, Info, Created At, Updated At
- Supports clicking on prompt IDs (currently just logs, can be extended for detail view)

## Usage

The component is integrated into the main application at:
- **Navigation**: Available in the left sidebar as "Prompts" (admin role required)
- **Routing**: Accessible via `?page=prompts` URL parameter
- **API**: Uses existing `getPromptsList` function from `networking.tsx`

## Props

```typescript
interface PromptsProps {
  accessToken: string | null
  userRole?: string
}
```

## Data Structure

The component expects prompts with the following structure:

```typescript
interface PromptItem {
  prompt_id?: string
  prompt_name: string | null
  prompt_info: Record<string, any>
  created_at?: string
  updated_at?: string
}
```

## Integration

The component is fully integrated into the main application:

1. **Left Navigation**: Added to `leftnav.tsx` with `FileTextOutlined` icon
2. **Main Routing**: Added to `page.tsx` with proper routing logic
3. **Permissions**: Restricted to admin roles (same as guardrails)

## Future Enhancements

- Add prompt detail view (similar to GuardrailInfoView)
- Add create/edit/delete functionality
- Add bulk operations
- Add search and filtering capabilities
- Add export functionality