# LiteLLM Chat UI Design System

## 1. Vision

A focused, professional chat interface that puts the conversation first. Inspired by the best of Microsoft Copilot, GitHub Copilot Chat, and modern SaaS tooling: clean surfaces, tight typography, restrained color, and purposeful motion. Every pixel earns its place

## 2. Design Principles

**Density over decoration.** Maximize usable space. No ornamental gradients, no heavy shadows, no gratuitous whitespace. The UI should feel like a precision tool, not a marketing page

**Quiet until needed.** Secondary controls (edit, delete, copy) appear on hover or focus. Primary actions (send, new chat) are always visible. Enterprise features (key rotation) surface only for entitled users

**System-native feel.** Use the OS font stack. Respect prefers-reduced-motion. Match native scrollbar behavior. The chat should feel like it belongs on the user's machine

**One layout, many views.** The sidebar + main area split is the single structural pattern. Sidebar switches between views (Chats, Apps, Credentials, API Keys, Usage) without page transitions. The main area hosts either the chat or a full-width panel

## 3. Layout

```
+--sidebar(260px)--+--------main-area--------+
| logo + collapse  | top-bar (model selector) |
| nav items        |                          |
| divider          |  content area            |
| view tabs        |  (chat / panel view)     |
| divider          |                          |
| [chats list]     |  input bar (chat only)   |
+------------------+--------------------------+
```

### Sidebar

- Width: 260px expanded, 56px collapsed
- Background: `bg-secondary` (the shadcn muted/secondary surface)
- Border right: `border` (1px, uses --border token)
- Collapse animation: 200ms cubic-bezier(0.4, 0, 0.2, 1)
- Nav items: 36px tall, 8px horizontal padding, radius-md, full-width
- Active nav item: `bg-accent` with `text-accent-foreground`
- Hover: `bg-accent/50`
- Icon size in nav: 16px. Label font: 14px/500

### Main area

- Background: `bg-background` (white in light mode)
- Top bar: 48px tall, bottom border, contains model selector left + settings right
- Content: flex column, fills remaining space
- Panel views (Apps, Credentials, Keys, Usage): max-width 800px, centered, 32px vertical padding, 24px horizontal

### Responsive behavior

- Below 768px: sidebar auto-collapses to icon rail
- Below 480px: sidebar becomes an overlay drawer
- Input bar sticks to bottom in all viewports

## 4. Color

Use the shadcn/Tailwind CSS variable system already defined in globals.css. No hardcoded hex values in components

| Token                      | Usage                                    |
| -------------------------- | ---------------------------------------- |
| `background`               | Main area, input surfaces                |
| `foreground`               | Primary text                             |
| `muted`                    | Sidebar background, disabled surfaces    |
| `muted-foreground`         | Secondary text, timestamps, placeholders |
| `primary`                  | Send button, active indicators, links    |
| `primary-foreground`       | Text on primary surfaces                 |
| `secondary`                | Sidebar surface, subtle backgrounds      |
| `border`                   | All 1px separators                       |
| `accent`                   | Hover states, active nav items           |
| `destructive`              | Delete actions, error states             |
| `card` / `card-foreground` | Elevated surfaces (modals, popovers)     |

### Accent colors for status

- Connected/success: `text-emerald-600`
- Warning/expiring: `text-amber-600`
- Error/expired: `text-destructive`
- Info/neutral: `text-muted-foreground`

## 5. Typography

Font stack: system default inherited from body (defined in globals.css, `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`)

| Element                 | Size | Weight | Line Height | Color              |
| ----------------------- | ---- | ------ | ----------- | ------------------ |
| Page heading (greeting) | 28px | 600    | 1.2         | `foreground`       |
| Section heading         | 18px | 600    | 1.3         | `foreground`       |
| Subsection heading      | 15px | 600    | 1.4         | `foreground`       |
| Body / message text     | 14px | 400    | 1.7         | `foreground`       |
| UI label (nav, button)  | 14px | 500    | 1           | varies             |
| Small label             | 13px | 400    | 1.4         | `muted-foreground` |
| Timestamp               | 11px | 400    | 1           | `muted-foreground` |
| Group header (Today)    | 11px | 600    | 1           | `muted-foreground` |
| Code (inline)           | 13px | 400    | 1.4         | `foreground`       |
| Code (block)            | 13px | 400    | 1.5         | `foreground`       |

Monospace stack for code: `ui-monospace, SFMono-Regular, 'SF Mono', Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace`

## 6. Spacing

Use Tailwind's spacing scale consistently. Avoid arbitrary pixel values

| Context                 | Value                                           |
| ----------------------- | ----------------------------------------------- |
| Sidebar padding (outer) | `p-2` (8px)                                     |
| Nav item padding        | `px-2.5 py-2` (10px, 8px)                       |
| Section divider margin  | `my-1 mx-2` (4px, 8px)                          |
| Content area padding    | `p-6` (24px) horizontal, `py-8` (32px) vertical |
| Card internal padding   | `p-3` to `p-4` (12-16px)                        |
| Message gap             | `gap-4` (16px) between messages                 |
| Button internal padding | `px-4 py-2` (16px, 8px) default                 |

## 7. Component Specifications

### 7.1 Chat Bubbles

**User bubble**

- Alignment: right-aligned, max-width 72%
- Background: `bg-muted` (light gray)
- Border radius: 16px (`rounded-2xl`)
- Padding: 10px 14px
- Font: 14px, line-height 1.6
- Edit button appears on hover to the left, icon-only, `text-muted-foreground`

**Assistant bubble**

- Alignment: left-aligned, max-width 80%
- No background (text renders directly)
- Markdown rendered with syntax highlighting for code blocks
- Copy button below, muted until hovered

**Tool call card**

- Collapsible (collapsed by default)
- Border: `border`, rounded-lg
- Background: `bg-muted/50`
- Header: tool icon + tool name, 13px/500
- Content sections labeled "Arguments" / "Result" in uppercase 11px/600

### 7.2 Input Bar

- Position: bottom of chat area, above-fold
- Container: `border rounded-xl` with inner padding
- Textarea: auto-growing, no visible border, inherits font
- Bottom row: model indicator left, send button right
- Send button: `bg-primary text-primary-foreground rounded-md px-4 py-1.5`
- Disabled send: `bg-muted text-muted-foreground cursor-not-allowed`
- Stop button (during streaming): circular, 32px, border only, square stop icon inside

### 7.3 Model Selector

- Trigger: ghost button in top bar showing model name + provider logo
- Dropdown: popover with search input + scrollable model list
- Selected model: checkmark icon, `text-primary`
- Comparison mode: up to 3 models, shown as pills

### 7.4 Sidebar Navigation

Each nav item is a full-width row:

```
[icon 16px] [label 14px/500]
```

- Default: `text-muted-foreground`, transparent background
- Hover: `bg-accent/50`
- Active: `bg-accent text-accent-foreground` with `font-medium`
- Collapsed sidebar: icon only, centered, with tooltip on hover

### 7.5 Conversation List

- Grouped by date: Today, Yesterday, Last 7 Days, Older
- Group headers: uppercase, 11px/600, `text-muted-foreground`, letter-spacing 0.04em
- Row: 34px min-height, `rounded-md`, 6px 8px padding
- Active row: `bg-accent`
- Hover row: `bg-accent/50`
- Action icons (edit, delete): appear on hover, opacity transition 150ms
- Delete confirmation: shadcn AlertDialog, not a popconfirm tooltip

### 7.6 Cmd+K Search

- Trigger: Ctrl/Cmd + K globally
- Dialog (shadcn Dialog, not Modal): 480px wide, centered vertically
- Search input at top with search icon prefix
- Results list: scrollable, max-height 320px
- Each result: message icon, truncated title, date badge right-aligned
- Click selects and closes

### 7.7 Data Tables (Keys, Credentials)

Use shadcn's table primitives (Table, TableHeader, TableBody, TableRow, TableCell) with Tailwind styling, not antd Table

- Border: `border rounded-lg overflow-hidden`
- Header row: `bg-muted/50`, uppercase 11px/600 labels
- Body rows: 44px min-height, `border-b last:border-0`
- Hover: `bg-accent/30`
- Cell text: 13-14px
- Action buttons: ghost/outline variant, icon-only where possible

### 7.8 Dialogs and Modals

Use shadcn Dialog component

- Overlay: black at 50% opacity
- Content: `bg-card`, rounded-xl, max-width varies (480px for search, 520px for forms)
- Header: title left, close X right
- Footer: action buttons right-aligned, primary action on the right
- Form inputs: shadcn Input with label above
- Spacing: 24px padding, 16px between form fields

### 7.9 Status Badges

Use shadcn Badge component

- Default (neutral): `bg-secondary text-secondary-foreground`
- Active/connected: green outline variant
- Warning/expiring: amber outline variant
- Error/expired: destructive variant
- Enterprise: primary variant with lock icon

### 7.10 Loading States

- Spinner: use a simple CSS spinner or Lucide `Loader2` icon with `animate-spin`, not antd Spin
- Skeleton: use shadcn Skeleton (pulsing gray bars)
- Typing indicator: three bouncing dots (keep existing CSS animation)
- Thinking placeholder: pulsing "Thinking..." text

### 7.11 MCP Server Cards (Apps Panel)

- Grid: 2-column on desktop, 1-column below 640px
- Card: `border rounded-lg` with padding 14px 16px
- Avatar: 38px, rounded-xl, either logo image or initial with hash-based color
- Hover: `bg-accent/30` transition 100ms
- Detail view: back button, larger avatar (64px), info table, tools list

### 7.12 Keys Panel

- Table with columns: Key Alias, Key (masked), Spend/Budget, Expires, Created
- Masked key format: `sk-...XXXX` (last 4 chars)
- Rotate button: outline variant, only rendered when `premiumUser === true`
- Non-premium users see a muted "Enterprise" badge where rotate would be
- Rotation modal: form with duration, key alias, max budget, grace period fields
- After rotation: show new key in a mono-font box with copy button

### 7.13 Usage Panel

- Time range selector: segmented control (7d / 30d / 90d)
- Stat cards row: 4 cards showing Total Spend, API Requests, Tokens, Success Rate
- Each card: `border rounded-lg p-4`, label above, large number below, subtitle
- Charts: simple inline bar charts using divs with percentage widths
- Colors: `bg-primary` for spend bars, `bg-chart-2` for request bars

### 7.14 Toggle/Switch (MCP Connect Picker)

Use shadcn Switch component

- Size: small (16px height)
- Track: `bg-input` unchecked, `bg-primary` checked
- Loading state: spinner overlaid

## 8. Interaction & Motion

| Interaction         | Timing      | Easing                       |
| ------------------- | ----------- | ---------------------------- |
| Hover background    | 150ms       | ease                         |
| Sidebar collapse    | 200ms       | cubic-bezier(0.4, 0, 0.2, 1) |
| Modal open/close    | 150ms       | ease-out / ease-in           |
| Icon opacity reveal | 150ms       | ease                         |
| Tooltip appear      | 200ms delay | ease                         |
| Button press        | 100ms       | ease                         |

All motion respects `prefers-reduced-motion: reduce` by collapsing to 0ms

## 9. Accessibility

- All interactive elements have visible focus rings (shadcn default: `ring-ring/50`)
- Keyboard navigation: Tab through nav items, Enter/Space to activate
- Cmd+K search is keyboard-accessible end-to-end
- Tooltip content duplicated in aria-label for icon-only buttons
- Color contrast: all text meets WCAG 2.1 AA (4.5:1 for body text, 3:1 for large text)
- Screen reader: conversation list items use role="listbox" + role="option"

## 10. shadcn Components Used

Components to install and use across the chat UI:

| Component   | Replaces (antd)      | Used in                       |
| ----------- | -------------------- | ----------------------------- |
| Button      | antd Button          | All buttons                   |
| Input       | antd Input           | Search, rename, form fields   |
| Dialog      | antd Modal           | Search, key rotation, delete  |
| AlertDialog | antd Popconfirm      | Delete confirmation           |
| Popover     | antd Popover         | Model selector, MCP picker    |
| Tooltip     | antd Tooltip         | Icon tooltips                 |
| Table\*     | antd Table           | Keys, credentials             |
| Badge       | antd Tag             | Status indicators             |
| Switch      | antd Switch          | MCP server toggles            |
| Select      | antd Select          | Time range, form selects      |
| Skeleton    | antd Skeleton        | Loading placeholders          |
| ScrollArea  | native overflow-auto | Conversation list, model list |
| Collapsible | antd Collapse        | Tool call cards               |
| Separator   | div with border      | Sidebar dividers              |
| Label       | (none)               | Form field labels             |
| Tabs        | custom tab buttons   | Apps panel tabs               |

\*For tables, use the shadcn table primitives (Table, TableHeader, TableRow, etc.) which are thin wrappers around native `<table>` with Tailwind classes, not a heavy data-table library

## 11. File Structure

```
src/components/chat/
  ChatPage.tsx          # Layout shell, routing, state orchestration
  ConversationList.tsx  # Sidebar chat list with date grouping
  ChatMessages.tsx      # Message rendering (user, assistant, tool)
  MCPAppsPanel.tsx      # Apps grid + detail view
  MCPConnectPicker.tsx  # MCP server toggle popover
  MCPCredentialsTab.tsx # OAuth credentials table
  KeysPanel.tsx         # API key viewing + rotation
  UsagePanel.tsx        # Spend/usage analytics
  types.ts              # Shared TypeScript types
  useChatHistory.ts     # Chat persistence hook
  design.md             # This file
```

## 12. Migration Notes

The migration from antd to shadcn is component-by-component. Each file should:

1. Replace antd imports with shadcn component imports + Lucide icons
2. Convert inline `style={{}}` objects to Tailwind classes where practical
3. Use CSS variables via Tailwind tokens (`bg-primary`, `text-muted-foreground`) instead of hardcoded hex
4. Keep the same prop interfaces and behavioral contracts
5. Preserve all keyboard shortcuts and accessibility features

antd's `@ant-design/icons` are replaced by Lucide React icons (already the shadcn default icon library per components.json). Mapping:

| antd icon           | Lucide equivalent |
| ------------------- | ----------------- |
| EditOutlined        | Pencil            |
| DeleteOutlined      | Trash2            |
| PlusOutlined        | Plus              |
| SearchOutlined      | Search            |
| MenuFoldOutlined    | PanelLeftClose    |
| MenuUnfoldOutlined  | PanelLeftOpen     |
| MessageOutlined     | MessageSquare     |
| AppstoreOutlined    | LayoutGrid        |
| KeyOutlined         | KeyRound          |
| LockOutlined        | Lock              |
| BarChartOutlined    | BarChart3         |
| SettingOutlined     | Settings          |
| ArrowLeftOutlined   | ArrowLeft         |
| DownOutlined        | ChevronDown       |
| CheckOutlined       | Check             |
| CopyOutlined        | Copy              |
| SyncOutlined        | RefreshCw         |
| ToolOutlined        | Wrench            |
| RightOutlined       | ChevronRight      |
| CheckCircleOutlined | CheckCircle       |
| UserOutlined        | User              |
| LinkOutlined        | Link              |
