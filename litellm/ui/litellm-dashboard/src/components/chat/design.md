# Chat UI Design Reference

Tactical guide for building chat UI features. Copy patterns exactly — don't improvise spacing, colors, or components.

The standalone `/ui/chat` workspace shares the dashboard’s navigation primitives, header, account menu, typography and theme tokens. The previous Moyai glass treatment made Chat look like a separate product. Keep Chat’s conversation tools and route-specific navigation inside the shared visual language.

> **Agents:** Read `AGENTS.md` alongside this file. It contains decision trees, the pre-commit checklist, and the rules that govern when to use each pattern here. This file is the _what_; `AGENTS.md` is the _when and how_.

---

## Stack

- **Next.js 16** App Router, TypeScript
- **Tailwind CSS v4** — shared primitives and utilities, with Chat-specific content layout in `moyai/moyai-chat.css` (home, composer and transcript)
- **shadcn/ui**: import from `@/components/ui/*`. Available today: `alert-dialog`, `badge`, `bubble`, `button`, `card`, `collapsible`, `dialog`, `input`, `input-group`, `label`, `popover`, `scroll-area`, `select`, `separator`, `sheet`, `skeleton`, `sonner`, `switch`, `table`, `tabs`, `textarea`, `tooltip`. Use these shared primitives when composing chat features
- **lucide-react** — only icon library, no emoji in UI
- The dashboard font, inherited from the app root. No separate font downloads or Chat palette

---

## Tokens

Never hardcode hex values. Use these CSS variables via Tailwind classes — all verified present in `src/app/globals.css`.

`ChatShell` and portalled dialogs carry `.chat-theme` only for Chat-specific message and editor aliases. Do not override the shared foreground, background, sidebar, primary or font tokens. Light and dark modes use the dashboard theme

### Surfaces

Use solid `bg-background` content and `bg-sidebar` navigation with `border-border` separators. No glass, dot/grid backgrounds, gradients or decorative shadows. The composer has a quiet border and a visible focus state. Model selection, attachments, tools and send remain in its action row

The welcome view contains a heading, composer and compact prompt shortcuts. Shortcuts fill the draft without submitting. Recent conversations belong in the sidebar, without an empty-count section. Show run and storage warnings when they apply, not as permanent idle helper text

The standalone `ChatModelPicker` exposes the account's chat and image-generation models from the composer action row. Its portalled popover carries `.chat-theme`; keep model names, capabilities and selection states readable in both themes. Attachment and MCP tools stay beside the composer.

The **Models** sidebar destination opens `/ui/chat/models` inside the same shell. It reuses the existing Model Hub catalog, filters, pricing and details in a read-only `chat` variant. Admins see the authenticated catalog and other users see the published catalog, matching the dashboard's existing access rules. The page uses `.moyai-models-page` for scrolling and `.chat-model-hub-card` for the bordered dashboard surface. Portalled model details carry `.chat-theme`; catalog management and other hub tabs stay in the dashboard.

Show available providers in a single-select dropdown beside model search. “All providers” clears the provider filter while keeping the other filters. The dropdown includes provider logos and uses the Chat theme in its portal. Provider logos reuse the existing asset mapping; white logo plates preserve the colors and contrast of the original assets in both themes. Unknown or failed logos fall back to a provider initial. The model list uses compact search and capability controls, quiet separators, and provider identities beside model names. Keep these styles scoped to Chat.

### Colors

| Token                         | Tailwind class                        | Use                                                        |
| ----------------------------- | ------------------------------------- | ---------------------------------------------------------- |
| `--background`                | `bg-background`                       | Main content area, input surfaces                          |
| `--foreground`                | `text-foreground`                     | Primary text                                               |
| `--card`                      | `bg-card`                             | Card, popover and dialog surfaces                          |
| `--muted`                     | `bg-muted`                            | Table header rows, subtle fills                            |
| `--muted-foreground`          | `text-muted-foreground`               | Secondary/helper text, timestamps                          |
| `--border`                    | `border-border` (or bare `border`)    | All 1px separators                                         |
| `--primary`                   | `bg-primary` / `text-primary`         | Send button, checkmarks, active links                      |
| `--destructive`               | `bg-destructive` / `text-destructive` | Delete, error actions                                      |
| `--sidebar`                   | `bg-sidebar`                          | **Left sidebar background — use this, not `bg-secondary`** |
| `--sidebar-foreground`        | `text-sidebar-foreground`             | Sidebar text                                               |
| `--sidebar-accent`            | `bg-sidebar-accent`                   | Active/hover nav item fill                                 |
| `--sidebar-accent-foreground` | `text-sidebar-accent-foreground`      | Active nav item text                                       |
| `--sidebar-border`            | `border-sidebar-border`               | Sidebar's own dividers/right border                        |

**Known gotcha:** generic `--accent`, `--secondary`, and `--muted` values may coincide in the dashboard theme. Always use the sidebar-specific tokens rather than relying on incidental contrast from a generic token. Rules:

- Sidebar container: `bg-sidebar`, never `bg-secondary` or `bg-accent`.
- Active/selected nav item: `bg-sidebar-accent text-sidebar-accent-foreground`, never `bg-accent` on its own inside the sidebar.
- Anywhere else `bg-accent`/`bg-muted` is used for hover/selected state, confirm the container isn't _also_ `bg-accent`/`bg-secondary`/`bg-muted` before shipping — check `globals.css` values, don't assume.

### Status colors (semantic — don't substitute)

| State               | Class                                                    | Use                         |
| ------------------- | -------------------------------------------------------- | --------------------------- |
| Success / connected | `text-emerald-600 dark:text-emerald-400`                 | Completed, connected, ready |
| Running / warning   | `text-amber-600 dark:text-amber-400`                     | In-flight, expiring         |
| Error               | `text-red-600 dark:text-red-400` (or `text-destructive`) | Failed, expired             |
| Info                | `text-muted-foreground`                                  | Informational               |

---

## Typography

```tsx
// Page/greeting heading
<header className="moyai-chat-home-header">
  <h1>What are we working on?</h1>
</header>

// Section heading (panel titles: "Your API Keys", "Your Usage")
<h2 className="text-base font-semibold tracking-tight">Your Usage</h2>

// Body text
<p className="text-sm text-foreground">Body</p>

// Secondary / helper text
<p className="text-xs text-muted-foreground">Helper</p>

// Table header
<th className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Col</th>

// Monospace (keys, IDs, JSON, tool args)
<span className="font-mono text-xs">sk-...abcd</span>
```

**Rules:**

- Use `tracking-tight` on headings
- Never set `text-foreground/50` or `text-foreground/70` for secondary text — use `text-muted-foreground`
- Monospace for: masked keys, conversation/tool-call IDs, tool arguments, JSON values

---

## Spacing

4px base grid.

| Context                            | Value                                       |
| ---------------------------------- | ------------------------------------------- |
| Sidebar outer padding              | `p-3`                                       |
| Nav item padding                   | `px-2.5 py-2`                               |
| Inline icon-to-label gap           | `gap-2.5` in nav items, `gap-1.5` elsewhere |
| Section divider margin             | `mx-2`                                      |
| Panel content padding              | `px-8 py-8`                                 |
| Card/bordered-div internal padding | `p-4`                                       |
| Dialog body padding                | `px-6 pb-6`                                 |

---

## Border Radius

```tsx
rounded - md; // 8px — buttons, inputs, nav items, bordered divs
rounded - lg; // 10px — panel cards, table containers
rounded - xl; // 12px — shared panels
rounded - lg; // 8px — user message bubbles
rounded - full; // pills, avatars, status dots
```

---

## Components

### Button

Always the shadcn `Button` — never a raw `<button>` with hand-rolled Tailwind, even for icon-only triggers. If it's borderline whether something is a "button" (e.g. a nav row, a suggestion pill), it still gets a `variant`.

```tsx
import { Button } from "@/components/ui/button"

<Button>Send</Button>
<Button variant="outline">Cancel</Button>
<Button variant="ghost">Nav item / low-emphasis</Button>
<Button variant="destructive">Delete</Button>
<Button variant="ghost" size="icon"><Settings className="size-4" /></Button>

// Loading
<Button disabled={loading}>
  {loading && <Loader2 className="size-4 animate-spin" />}
  Save
</Button>
```

Sizes: `default` (h-9), `sm` (h-8), `xs` (h-6), `lg` (h-10), `icon` (size-9), `icon-sm` (size-8), `icon-xs` (size-6).

### Sidebar nav item

Use `Sidebar`, `SidebarContent`, `SidebarFooter` and `SidebarMenuButton` from `components/shared/Sidebar`. Both workspaces use `SidebarBrand`, `ViewSwitcher` and `DashboardHeader`. Chat has a compact account menu with its existing identity, Settings and Sign out actions. Theme lives in the shared top bar. Gateway account preferences and admin controls stay in the gateway. Keep the shared 280px expanded and 72px collapsed widths, 56px header and 768px mobile breakpoint

Use one New Chat action for the blank composer; do not duplicate it with a Chats destination. Chat retains Models, Integrations, Workspace and conversation history. Chat has no gateway-wide header search or worker selector. Search chats in the sidebar owns Cmd/Ctrl+K, including when the sidebar is collapsed or on mobile. Mobile navigation uses the same Sheet treatment as the dashboard

The conversation title appears in the shared header. Do not repeat it in a second content header. Routed pages keep their content-specific headings and actions

### Routed panel (Integrations, Credentials, API Keys, Usage)

```tsx
<div className="flex-1 min-h-0 overflow-auto w-full py-8 px-8">
  <PanelComponent />
</div>
```

No max-width cap on panel content — matches the rest of the dashboard's own pages (verified: `(dashboard)/api-keys/ApiKeysDashboard.tsx` has no width cap). A narrow `max-w-[800px]` here previously made wide tables (Keys) look cramped on real monitors; don't reintroduce it.

---

## Icons

`lucide-react` only. `size-4` (16px) inline/nav, `size-5` (20px) standalone actions, `size-3.5` (14px) inside badges/small buttons.

---

## Dos and Don'ts

| Do                                                                    | Don't                                                                               |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `Button` (any variant) for every clickable control                    | Raw `<button>` with hand-rolled Tailwind                                            |
| `bg-sidebar` / `bg-sidebar-accent` for sidebar surfaces               | `bg-secondary` / `bg-accent` for sidebar surfaces (identical values, zero contrast) |
| `Tabs variant="line"` for section tabs                                | Default `Tabs` variant + manual `border-b-2` overrides                              |
| Skeleton loaders for panel/list content                               | Spinner or blank screen while fetching                                              |
| `text-muted-foreground` for secondary text                            | `text-foreground/50`, `text-foreground/60`, `text-foreground/70`                    |
| `AlertDialog` for destructive confirmations                           | Instant delete, or a hand-rolled confirm                                            |
| Verify a prop/variant exists in `components/ui/*.tsx` before using it | Assume a shadcn prop exists because another shadcn app has it                       |
| `toast` from `@/lib/toast`, backed by the root sonner instance        | Introduce a second, competing toast library                                         |

---

## Known gaps (tracked, not blockers)

Card, Textarea and sonner are installed. Earlier versions of this document listed them as missing and encouraged parallel implementations; use the shared owners above

Shared sidebar primitives are available in `components/shared/Sidebar`. Use them instead of hand-writing a separate rail

## File Structure

```
src/components/chat/
  ChatShell.tsx          # Desktop navigation / mobile Sheet, shared across all /chat/* routes
  ConversationList.tsx   # Date-grouped chat list and conversation search
  ChatMessages.tsx       # Shared list and single-message rendering (user, assistant, tool)
  MCPAppsPanel.tsx       # Integrations grid + detail view
  MCPConnectPicker.tsx   # MCP server toggle popover
  MCPCredentialsTab.tsx  # OAuth credentials table
  KeysPanel.tsx          # API key viewing + rotation
  UsagePanel.tsx         # Spend/usage analytics
  types.ts               # Shared TypeScript types
  useChatHistory.ts      # Chat persistence hook
  design.md              # This file
  AGENTS.md              # When/how to apply this file
../../contexts/ChatShellContext.tsx  # Cross-route state: MCP server selection, conversation history
../../app/chat/                      # Route tree: page.tsx (chats), layout.tsx (auth gate + shell),
                                      # integrations/, credentials/, api-keys/, usage/
```
