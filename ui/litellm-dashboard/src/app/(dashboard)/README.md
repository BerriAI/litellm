# Contributing to the LiteLLM UI

The LiteLLM UI is currently being refactored/rewritten to reduce development friction. Please read this document to understand what's expected for new contributions.

The project follows strict NextJS file structure. All pages on the site (determined by the sidebar) are contained in their own folder, and routing is automatically handled by NextJS based on the file structure. 

For example, NextJS will automatically render the admin settings page when the user visits `/settings/admin-settings`

```
.
├── settings
    ├── admin-settings
        └── page.tsx
```

You can use parenthesis around directory names to hide them from the user route, for example `(dashboard)`, while still getting the benefits of `layout` and file structure.

### File Structure
Every page must follow the following file structure pattern.
```
├── teams
│   ├── TeamsView.tsx
│   ├── components
│   │   ├── TeamsFilters.tsx
│   │   ├── TeamsHeaderTabs.tsx
│   │   ├── TeamsTable
│   │   │   ├── ModelsCell.tsx
│   │   │   └── TeamsTable.tsx
│   │   └── modals
│   │       ├── CreateTeamModal.tsx
│   │       └── DeleteTeamModal.tsx
│   ├── hooks
│   │   └── useFetchTeams.ts
│   └── page.tsx
```

### Component  Files

All component files should ideally be as dumb as possible. Their only job should be to take the data they need from hooks or props and render them to the UI. If a component file becomes too large (over `300` lines or so), **please break it down** into smaller components.

A component should only be placed where it will be used. For example, if a component will only be used by the `teams` page, it should belong in the `teams/components` folder. 

**Common components should be moved to the lowest common ancestor components folder.**

### Hooks

All hooks should be pure `.ts` files in a dedicated `hooks` folder unless they serve as context managers.

**Common hooks should be moved to the lowest common ancestor hooks folder.**

### Utils

Any pure `.ts` functions that you need in order to process data should be placed in a local `utils.ts` file.
