# Data Retention Policy

## LiteLLM Cloud

### Purpose
This policy outlines the requirements and controls/procedures LiteLLM Cloud has implemented to manage the retention and deletion of customer data.

### Policy

For Customers
1. Active Accounts

- Customer data is retained for as long as the customer’s account is in active status. This includes data such as prompts, generated content, logs, and usage metrics. By default, we do not store the message / response content of your API requests or responses. Cloud users need to explicitly opt in to store the message / response content of your API requests or responses.

2. Voluntary Account Closure

- Data enters an “expired” state when the account is voluntarily closed.
- Expired account data will be retained for 30 days (adjust as needed).
- After this period, the account and all related data will be permanently removed from LiteLLM Cloud systems.
- Customers who wish to voluntarily close their account should download or back up their data (manually or via available APIs) before initiating the closure process.

3. Involuntary Suspension

- If a customer account is involuntarily suspended (e.g., due to non-payment or violation of Terms of Service), there is a 14-day (adjust as needed) grace period during which the account will be inaccessible but can be reopened if the customer resolves the issues leading to suspension.
- After the grace period, if the account remains unresolved, it will be closed and the data will enter the “expired” state.
- Once data is in the “expired” state, it will be permanently removed 30 days (adjust as needed) thereafter, unless legal requirements dictate otherwise.

4. Manual Backup of Suspended Accounts

- If a customer wishes to manually back up data contained in a suspended account, they must bring the account back to good standing (by resolving payment or policy violations) to regain interface/API access.
- Data from a suspended account will not be accessible while the account is in suspension status.
- After 14 days of suspension (adjust as needed), if no resolution is reached, the account is closed and data follows the standard “expired” data removal timeline stated above.

5. Custom Retention Policies

- Enterprise customers can configure custom data retention periods based on their specific compliance and business requirements.
- Available customization options include:
  - Adjusting the retention period for active data (0-365 days)
- Custom retention policies must be configured through the LiteLLM Cloud dashboard or via API


### Protection of Records

- LiteLLM Cloud takes measures to ensure that all records under its control are protected against loss, destruction, falsification, and unauthorized access or disclosure. These measures are aligned with relevant legislative, regulatory, contractual, and business obligations.
- When working with a third-party CSP, LiteLLM Cloud requests comprehensive information regarding the CSP’s security mechanisms to protect data, including records stored or processed on behalf of LiteLLM Cloud.
- Cloud service providers engaged by LiteLLM Cloud must disclose their safeguarding practices for records they gather and store on LiteLLM Cloud’s behalf.

