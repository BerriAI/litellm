---
sidebar_position: 2
---

# Quickstart: xct-home

xct-home is the public landing page. It renders **anonymously** — no
user login — using `/.well-known/xct-capabilities`.

## What you'll build

- Static / SSR landing page that lists what xct-litellm offers
- A "Models we support" grid
- A "Featured agents" carousel
- A "Skills marketplace" section
- All without any OAuth

## 1. The endpoint

```http
GET /.well-known/xct-capabilities
→ 200 application/json
```

No auth header required. Returns `PublicCapabilitiesResponse` —
only entities the operator marked `is_public`:

- `litellm.public_model_groups` (config flag)
- `litellm.public_agent_groups` (config flag)
- `litellm.public_mcp_groups` (config flag)
- `is_public=true` xct-skills rows

5-minute cache. Stripped of credential / topology fields.

## 2. SSR fetch

```tsx
// pages/index.tsx (Next.js example)
import type { GetStaticProps } from "next";

export const getStaticProps: GetStaticProps = async () => {
  const resp = await fetch("https://api.xct.test/.well-known/xct-capabilities");
  if (!resp.ok) throw new Error("Failed to load capabilities");
  return { props: await resp.json(), revalidate: 300 };
};

export default function Home(props: any) {
  return (
    <>
      <Hero />
      <ModelGrid models={props.models} />
      <AgentCarousel agents={props.agents} />
      <SkillMarketplace skills={props.skills} />
    </>
  );
}
```

## 3. Model card

The model summary already carries everything for the card:

```tsx
function ModelCard({ model }) {
  return (
    <div>
      <h3>{model.id}</h3>
      <span>{model.provider}</span>
      <p>{model.context_window?.toLocaleString()} context tokens</p>
      <div>
        {model.capabilities.vision && <Tag>vision</Tag>}
        {model.capabilities.function_calling && <Tag>tools</Tag>}
        {model.capabilities.prompt_caching && <Tag>caching</Tag>}
      </div>
    </div>
  );
}
```

## 4. Skill marketplace

```tsx
function SkillMarketplace({ skills }) {
  return (
    <Grid>
      {skills.map(s => (
        <SkillCard
          key={s.skill_id}
          title={s.display_title}
          version={s.version}
          category={s.category}
          description={s.description}
        />
      ))}
    </Grid>
  );
}
```

If a visitor clicks **Try this skill**, you bounce them to the
xct-chat OAuth flow — keep `/v1/capabilities` (the auth'd version) for
the post-login experience.

## What's NOT in the public response

- caller block (no key/team/user IDs)
- MCP `transport`, `auth_type`, `access_groups` (deployment topology leak)
- Private agents / private MCPs / private skills
- Models not in `litellm.public_model_groups`

## 5. (Optional) Anonymous "browse public agents" details

```http
GET /v1/agents/{agent_id}/.well-known/agent-card.json
```

When the agent is public, this card is reachable anonymously (S3-01
ACL allows public agents through). Use it to render the agent detail
page before the user logs in.

## Caching

The proxy already caches the public response for 5 minutes
(`LITELLM_PUBLIC_CAPABILITIES_CACHE_TTL`). Add your own SSG / ISR cache
on top for cheap.
