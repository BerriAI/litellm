---
sidebar_position: 5
---

# OAuth PKCE in React

```tsx
// LoginButton.tsx
import { beginPkce } from "@xct/litellm-sdk";

export function LoginButton() {
  const start = async () => {
    const session = await beginPkce({
      baseUrl: import.meta.env.VITE_PROXY_URL,
      clientId: import.meta.env.VITE_OAUTH_CLIENT_ID,
      redirectUri: window.location.origin + "/oauth/callback",
      scope: "read write",
    });
    window.location.href = session.authorizeUrl;
  };
  return <button onClick={start}>Log in</button>;
}
```

```tsx
// OAuthCallback.tsx — render at /oauth/callback
import { useEffect } from "react";
import { completePkce } from "@xct/litellm-sdk";
import { useNavigate, useSearchParams } from "react-router-dom";

export function OAuthCallback() {
  const [params] = useSearchParams();
  const nav = useNavigate();
  useEffect(() => {
    (async () => {
      const code = params.get("code");
      if (!code) return;
      const tokens = await completePkce(import.meta.env.VITE_PROXY_URL, {
        code,
        state: params.get("state") ?? undefined,
      });
      sessionStorage.setItem("xct_access_token", tokens.access_token);
      sessionStorage.setItem("xct_refresh_token", tokens.refresh_token ?? "");
      nav("/");
    })();
  }, [params, nav]);
  return <div>Signing you in…</div>;
}
```

Notes:
- `beginPkce` writes the code_verifier + state to **sessionStorage**, not
  localStorage (proxy convention — no credentials in localStorage)
- `completePkce` validates state matches and clears the session entry
- If user opens the OAuth tab and never comes back, the sessionStorage
  entry dies with the tab — no leak
