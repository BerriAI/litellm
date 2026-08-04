# Merge AD directory search and manual email into one field

## Problem

The Invite User modal (`CreateUserButton.tsx`) currently renders two separate inputs for what is conceptually one piece of data:

- "Search AD Users" — a directory-search combobox (`AdDirectoryUserSearch`), shown only when `MICROSOFT_DIRECTORY_SEARCH_ENABLED` is on. Selecting a result calls an `onSelectUser` callback that writes into a second field via `form.setFieldsValue`.
- "User Email" — a plain antd `Input`/`TextInput` bound to the `user_email` form field, always rendered.

This reads as two unrelated inputs even though they resolve to the same underlying value, and a user could type into "User Email" while "Search AD Users" sits above showing nothing selected.

## Goal

Collapse both into a single `Form.Item name="user_email"` field. Which control renders inside it is chosen by `uiSettings.MICROSOFT_DIRECTORY_SEARCH_ENABLED`:

- **Enabled:** the field itself is the directory-search combobox.
- **Disabled or not yet loaded:** the field is the existing plain `Input`/`TextInput`, unchanged from today.

## Behavior

### Directory search enabled

- Label stays "User Email" (not "Search AD Users") in both cases, so the form's visible contract doesn't change based on config. Placeholder text is "Search by name or email".
- Typing filters the directory via the existing debounced `directoryUsersSearchCall`, same UX as today's combobox (loading text, "no users found", error surfacing).
- Selecting a result from the dropdown sets the field value to that user's email.
- Typed text that is never selected from the dropdown does **not** commit to the field value. This isn't extra logic — it's how the underlying `PaginatedSearchSelect`/Base UI `Combobox` already behaves (`onValueChange` only fires on item selection). The practical effect: a user must pick from the directory to produce a value.
- The `user_email` field gets a `required` validation rule with message "Select a user from the directory search results", so submitting without a selection is blocked client-side with an inline error, consistent with other required `Form.Item`s in this form.

### Directory search disabled (or `uiSettings` still loading)

- Renders the existing plain `Input` (`TextInput` in embedded mode) exactly as today.
- No required-field rule is added here — manual-entry mode keeps its current behavior (empty submissions are only caught server-side). This is a deliberate scope limit: the ask is to merge the fields, not to change validation behavior for proxies that don't use directory search.

## Component contract change: `AdDirectoryUserSearch`

Before:
```ts
interface AdDirectoryUserSearchProps {
  accessToken: string;
  onSelectUser: (user: DirectoryUser) => void;
}
```

After:
```ts
interface AdDirectoryUserSearchProps {
  accessToken: string;
  value?: string;
  onChange?: (email: string) => void;
  disabled?: boolean;
}
```

- Becomes a standard antd-compatible controlled field (`value`/`onChange`), so it drops directly into `Form.Item` with no glue code in `CreateUserButton.tsx`.
- Internal search state (`options`, `isLoading`, `error`, the out-of-order-response guard via `requestIdRef`) is unchanged.
- The `usersByEmail` lookup map is no longer needed for handing back a full `DirectoryUser` — only the selected email travels outward via `onChange`. This simplifies the component: search results resolve straight to `{ label, value: email, sublabel }` and selection just forwards `value` (the email string) to `onChange`.
- Drops its own internal `<Label htmlFor={INPUT_ID}>Search AD Users</Label>` entirely. The label is now owned by the wrapping `Form.Item label="User Email"` in `CreateUserButton.tsx`; the component renders only the `PaginatedSearchSelect` itself, so there's exactly one label per field instead of a doubled-up label.

## `CreateUserButton.tsx` changes

- Delete the separate `directoryUserSearchItem` and `userEmailFormItem` JSX blocks.
- Replace with a single `Form.Item` whose child is chosen by a ternary on `uiSettings?.MICROSOFT_DIRECTORY_SEARCH_ENABLED`:
  ```tsx
  <Form.Item
    label="User Email"
    name="user_email"
    rules={
      uiSettings?.MICROSOFT_DIRECTORY_SEARCH_ENABLED
        ? [{ required: true, message: "Select a user from the directory search results" }]
        : []
    }
  >
    {uiSettings?.MICROSOFT_DIRECTORY_SEARCH_ENABLED ? (
      <AdDirectoryUserSearch key={directorySearchResetSignal} accessToken={accessToken} />
    ) : isEmbedded ? (
      <TextInput placeholder="" />
    ) : (
      <Input />
    )}
  </Form.Item>
  ```
- Used identically in both the embedded and modal render paths (the component's only two render branches, both currently duplicating this same pair of fields).
- The existing `directorySearchResetSignal` + `key` remount trick is kept, scoped to the `AdDirectoryUserSearch` branch only — it clears stale search results/error text when the modal reopens. This is independent of `form.resetFields()`, which handles the actual `user_email` value.

## Testing

- `AdDirectoryUserSearch.test.tsx`: update every test's prop usage from `onSelectUser={fn}` to `value`/`onChange={fn}`; the "selects a user" test now asserts `onChange` was called with the plain email string (`"alice@example.com"`), not a `DirectoryUser` object. Existing coverage (debounce, stale-response discard, error surfacing, remount-clears-state) carries over unchanged since none of it depends on the removed `onSelectUser` shape.
- `CreateUserButton.test.tsx` / `.integration.test.tsx` (classify per the tier split in `ui/litellm-dashboard/CLAUDE.md` — this form's field-selection logic is a rendering/wiring concern, so it belongs in the integration tier): add/extend cases for:
  - Directory search enabled: selecting a directory result populates `user_email` and submit succeeds; submitting without a selection shows the required-field error and does not call `userCreateCall`.
  - Directory search disabled: typing an arbitrary email and submitting still works with no required-field error, matching current behavior.
  - Reopening the modal after directory search was used clears the combobox's prior search text/results.

## Out of scope

- No change to server-side validation of `user_email`.
- No change to manual-entry-mode validation (deliberately left as-is per the "leave as-is" decision above).
- No change to other consumers of these fields — `AdDirectoryUserSearch` has exactly one consumer today (`CreateUserButton.tsx`).
