# How Budgets Work in LiteLLM

LiteLLM's budget system is hierarchical and flexible, allowing you to control spending at multiple organizational levels. Here's how budgets work from top to bottom:

### Budget Hierarchy

**1. Global Proxy Level**
- Applies to all requests across the entire proxy server
- Set in `config.yaml` using `max_budget` and `budget_duration`
- Every API call counts toward this global budget

**2. Organization/Team Level**
- Teams act as organizational containers with shared budgets
- Multiple users can belong to a team and share a team budget
- All keys assigned to a team use the team's budget, not individual user budgets
- Teams can have multiple members with different roles (admin, user)
- When a team budget is exceeded, all team members' requests fail

**3. Team Member Level**
- Individual users within a team can have their own spending limits (`max_budget_in_team`)
- A team member's budget is a subset of the team's total budget
- Useful for controlling individual spending within a shared team allocation
- Example: Team has $1,000/month budget, but a specific member has $100/month limit

**4. Internal User Level**
- Personal budgets for users who own API keys
- Applied when a key does NOT have a `team_id`
- Each user has their own spend tracking across all their keys
- If a user has multiple keys, the budget applies to all of them combined
- Example: User "krrish@berri.ai" has a $50/month personal budget

**5. Virtual Key Level**
- Individual API keys can have their own budgets
- Most granular level of control
- Useful for isolating spending per application or service
- Example: "Production key" has $500/month, "Testing key" has $10/month
- Response returns `x-litellm-key-remaining` headers showing remaining budget

**6. Model-Specific Budgets** ✨ (Enterprise)
- Set different budget limits per model on a single key
- Example: GPT-4o limited to $0.0001/day, GPT-4o-mini limited to $10/month
- Allows fine-grained control over expensive vs. budget-friendly models

**7. Customer/End-User Level**
- Budget customers who pass a `user` parameter in API calls
- No key creation needed - just pass `user="customer_id"` in requests
- Perfect for SaaS applications with multiple end customers
- Each customer gets their own spend tracking and budget limits
- Example: Free-tier customers limited to $0.0001/month

### Budget Precedence (Which Budget Applies?)

When multiple budgets exist, LiteLLM applies them in this order:

1. **Team Budget** (if key has `team_id`) - Takes precedence over user budget
2. **Team Member Budget** (if user is in a team with `max_budget_in_team`)
3. **Internal User Budget** (if key has `user_id` but no `team_id`)
4. **Virtual Key Budget** (if key has `max_budget`)
5. **Model-Specific Budget** (if set for that model)
6. **Global Proxy Budget** (applies to everything)

### Spend Tracking

- **Real-time Tracking**: Costs are calculated per API call and stored immediately in the database
- **Database Storage**: All spend is persisted in `LiteLLM_VerificationToken` table for auditing
- **Automatic Aggregation**: Spend automatically rolls up from keys → users → teams
- **No Double Counting**: If a key belongs to a team, team spend includes that key's spend (not counted twice)

### Budget Resets

- **Automatic Resets**: Set `budget_duration` (e.g., "10d", "30d", "1mo") and budgets reset automatically
- **Reset Timing**: By default, LiteLLM checks for resets every 10 minutes to minimize database calls
- **Custom Reset Frequency**: Configure with `proxy_budget_rescheduler_min_time` and `proxy_budget_rescheduler_max_time`
- **No Duration = No Reset**: If `budget_duration` is not set, the budget never resets (cumulative)

### Common Budget Scenarios

**Scenario 1: SaaS with Multiple Customers**
- Set `max_end_user_budget: 0.0001` globally
- Each customer gets their own budget tracked by the `user` field in requests
- No keys needed per customer

**Scenario 2: Multi-Team Organization**
- Create separate teams for different departments
- Set team budgets (e.g., Team A: $1000/mo, Team B: $500/mo)
- Team budgets override individual user budgets automatically

**Scenario 3: Development Environment Control**
- Create keys with low budgets for dev/testing
- Create separate keys with higher budgets for production
- Both keys can belong to the same user but have different limits

**Scenario 4: Expensive Model Gating**
- Use model-specific budgets to limit GPT-4 usage
- Set GPT-3.5-turbo with a higher budget for routine tasks
- Prevent accidental expensive calls from using all budget

### What Happens When Budget Exceeds?

When a budget is exceeded:
- **Hard Budget** (default): Request fails immediately with error `ExceededBudget`
- **Soft Budget** (if configured): Request succeeds but warning is logged
- **Error Response**: Returns `401` with message showing current spend vs. max budget
- **Example**: `"Budget has been exceeded: User ishaan3 has exceeded their budget. Current spend: 0.0008869999999999999; Max Budget: 0.0001"`
