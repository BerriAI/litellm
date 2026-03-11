# Tag-Based Policy Attachments

Apply guardrail policies automatically to any key or team that has a specific tag. Instead of attaching policies one-by-one, tag your keys and let the policy engine handle the rest.

**Example:** Your security team requires all healthcare-related keys to run PII masking and PHI detection. Tag those keys with `health`, create a single tag-based attachment, and every matching key gets the guardrails automatically.

## 1. Create a Policy with Guardrails

Navigate to **Policies** in the left sidebar. You'll see a list of existing policies along with their guardrails.

![Policies list page showing existing policies and the + Add New Policy button](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/d7aa1e1f-011e-40bf-a356-6dfe9d5d54f1/ascreenshot_8db95c231a7f4a79a36c2a98ba127542_text_export.jpeg)

Click **+ Add New Policy**. In the modal, enter a name for your policy (e.g., `high-risk-policy2`). You can also type to search existing policy names if you want to reference them.

![Create New Policy modal — enter the policy name and optional description](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/18f1ff69-9b83-4a98-9aad-9892a104d3ff/ascreenshot_1c6b85231cad4ec695750b53bbbda52c_text_export.jpeg)

Scroll down to **Guardrails to Add**. Click the dropdown to see all available guardrails configured on your proxy — select the ones this policy should enforce.

![Guardrails to Add dropdown showing available guardrails like OAI-moderation, phi-pre-guard, pii-pre-guard](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/55cedad7-9939-44a1-8644-a184cde82ab7/ascreenshot_eab4e55b82b8411893eccb6234d60b82_text_export.jpeg)

After selecting your guardrails, they appear as chips in the input field. The **Resolved Guardrails** section below shows the final set that will be applied (including any inherited from a parent policy).

![Selected guardrails shown as chips: testing-pl, phi-pre-guard, pii-pre-guard. Resolved Guardrails preview below.](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/c06d5b08-1c85-4715-b827-3e6864880428/ascreenshot_7a082e55f3ad425f9009346c68afae23_text_export.jpeg)

Click **Create Policy** to save.

![Click Create Policy to save the new policy](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/7e6eae64-4bba-4d72-b226-d1308ac576a8/ascreenshot_22d0ed686c594221bbbd2f40df214d75_text_export.jpeg)

## 2. Add a Tag Attachment for the Policy

After creating the policy, switch to the **Attachments** tab. This is where you define *where* the policy applies.

![Switch to the Attachments tab — shows the attachment table and scope documentation](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/871ae6d9-16d1-44e2-baf2-7bb8a9e72087/ascreenshot_76e124619d70462ea0e2fbb46ded1ac9_text_export.jpeg)

Click **+ Add New Attachment**. The Attachments page explains the available scopes: Global, Teams, Keys, Models, and **Tags**.

![Attachments page showing scope types including Tags — click + Add New Attachment](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/d45ab8bc-fc1e-425b-8a3f-44d18df810ec/ascreenshot_425824030f3144b7ab3c0ac570349b00_text_export.jpeg)

In the **Create Policy Attachment** modal, first select the policy you just created from the dropdown.

![Select the policy to attach from the dropdown (e.g., high-risk-policy2)](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/e0dcac40-e39c-4a6a-9d9c-4bbb9ec0ee91/ascreenshot_445b19894e0b466196a13e20c8e67f2d_text_export.jpeg)

Choose **Specific (teams, keys, models, or tags)** as the scope type. This expands the form to show fields for Teams, Keys, Models, and Tags.

![Select "Specific" scope type to reveal the Tags field](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/f685e02a-e22e-4c6c-9742-d5268746214b/ascreenshot_14d63d9d06dd4fc7854cfeb5e8d9ef85_text_export.jpeg)

Scroll down to the **Tags** field and type the tag to match — here we enter `health`. You can enter any string, or use a wildcard pattern like `health-*` to match all tags starting with `health-` (e.g., `health-team`, `health-dev`).

![Tags field with "health" entered. Supports wildcards like prod-* matching prod-us, prod-eu.](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/14581df7-732c-4ea5-b36d-58270b00e92c/ascreenshot_e734c81418f046549b61a84b9d352a29_text_export.jpeg)

## 3. Check the Impact of the Attachment

Before creating the attachment, click **Estimate Impact** to preview how many keys and teams would be affected. This is your blast-radius check — make sure the scope is what you expect before applying.

![Click Estimate Impact — the tag "health" is entered and ready to preview](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/6ccb81d7-3d11-48b0-b634-fc4d738aa530/ascreenshot_2eb89e6ff13a4b12b61004660a36c30c_text_export.jpeg)

The **Impact Preview** appears inline, showing exactly how many keys and teams would be affected. In this example: "This attachment would affect **1 key** and **0 teams**", with the key alias `hi` listed.

![Impact Preview showing "This attachment would affect 1 key and 0 teams." Keys: hi](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/8834d85a-2c15-48dd-8d6b-810cf11ee5c4/ascreenshot_d814b42ca9f34c23b0c2269bfa3e64fb_text_export.jpeg)

Once you're satisfied with the impact, click **Create Attachment** to save.

![Click Create Attachment to finalize](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/4a8918f2-eedb-4f49-a53b-4e46d0387d2a/ascreenshot_b08d490d836d4f46b4e5cbb14f61377a_text_export.jpeg)

The attachment now appears in the table with the policy name `high-risk-policy2` and tag `health` visible.

![Attachments table showing the new attachment with policy high-risk-policy2 and tag "health"](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/45867887-0aec-44a4-963b-b6cc6c302e3e/ascreenshot_981caeff98574ec89a8a53cd295e5043_text_export.jpeg)

## 4. Create a Key with the Tag

Navigate to **Virtual Keys** in the left sidebar. Click **+ Create New Key**.

![Virtual Keys page showing existing keys — click + Create New Key](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/4c1f9448-e590-4546-9357-6f68aa395b27/ascreenshot_4a7bc5be9e4347f3a9fe46f78d938d7c_text_export.jpeg)

Enter a key name and select a model. Then expand **Optional Settings** and scroll down to the **Tags** field.

![Create New Key modal — enter the key name](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/f84f7a2b-8057-4926-9f80-d68e437c77cf/ascreenshot_a277c8611b6e41059663b0759cd85cab_text_export.jpeg)

In the **Tags** field, type `health` and press Enter. This is the tag the policy engine will match against.

![Tags field in key creation — type "health" to add the tag](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/3ad3bf10-76d2-4f15-9a66-ed6c99bb25c4/ascreenshot_8a8773fb65fc49329cb1716da92b2723_text_export.jpeg)

The tag `health` now appears as a chip in the Tags field. Confirm your settings look correct.

![Tags field showing "health" selected with a checkmark](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/de3e58a9-6013-4d0c-882e-5517ea286684/ascreenshot_c7eef1736fce4aa894ac3b118b3800a2_text_export.jpeg)

Click **Create Key** at the bottom of the form.

![Click Create Key to generate the new virtual key with the health tag](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/51d419ea-ee80-4e24-8e93-b99a844881bc/ascreenshot_097d4564289943a88e30b5d2e3eab262_text_export.jpeg)

A dialog appears with your new virtual key. Click **Copy Virtual Key** — you'll need this to test in the next step.

![Save your Key dialog — click Copy Virtual Key to copy it to clipboard](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/e87a0cc1-4d12-4066-bfa2-973159808fd1/ascreenshot_7b616a7291d0497a9c61bdcdb59394d7_text_export.jpeg)

## 5. Test the Key and Validate the Policy is Applied

Navigate to **Playground** in the left sidebar to test the key interactively.

![Navigate to Playground from the sidebar](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/e6f8a3ee-e9e8-4107-93d1-bfca734c5ce9/ascreenshot_539bde38abe646e49148a912fff2d257_text_export.jpeg)

Under **Virtual Key Source**, select "Virtual Key" and paste the key you just copied into the input field.

![Paste the virtual key into the Playground configuration](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/a6612c4a-d499-4e54-8019-f54fde674ad9/ascreenshot_e85ebb9051554594bab0da57823fafad_text_export.jpeg)

Select a model from the **Select Model** dropdown.

![Select a model (e.g., bedrock-claude-opus-4.5) from the dropdown](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/325e330f-3eff-4c5e-b177-21916138a2f5/ascreenshot_693478f89c034e949e08f3ed0dd05120_text_export.jpeg)

Type a message and press Enter. If a guardrail blocks the request, you'll see it in the response. In this example, the `testing-pl` guardrail detected an email pattern and returned a 403 error — confirming the policy is working.

![Guardrail in action — the request was blocked with "Content blocked: email pattern detected"](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/2cf16809-d2e5-4eae-a7dd-6a16dfcca7ce/ascreenshot_727d7d4ed20b4a52b2b41e39fd36eccb_text_export.jpeg)

**Using curl:**

You can also verify via the command line. The response headers confirm which policies and guardrails were applied:

```bash
curl -v http://localhost:4000/chat/completions \
  -H "Authorization: Bearer <your-tagged-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "say hi"}]
  }'
```

Check the response headers:

```
x-litellm-applied-policies: high-risk-policy2
x-litellm-applied-guardrails: pii-pre-guard,phi-pre-guard,testing-pl
x-litellm-policy-sources: high-risk-policy2=tag:health
```

| Header | What it tells you |
|--------|-------------------|
| `x-litellm-applied-policies` | Which policies matched this request |
| `x-litellm-applied-guardrails` | Which guardrails actually ran |
| `x-litellm-policy-sources` | **Why** each policy matched — `tag:health` confirms it was the tag |
