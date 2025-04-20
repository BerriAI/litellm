
import Image from '@theme/IdealImage';

# SCIM with Litellm

Using SCIM with LitELLM allows you to sync users, groups and group memberships on LiteLLM with your IDP (SSO) Provider. 

The key benefits of using SCIM with LiteLLM are:
- Users are automatically provisioned, deprovisioned, and updated on LiteLLM when they are added to a group in your IDP.
- Groups are automatically provisioned, deprovisioned, and updated on LiteLLM when they are created in your IDP.
- Group memberships are automatically provisioned, deprovisioned, and updated on LiteLLM when they are added to a group in your IDP.

## 1. Create a SCIM Application in your IDP

## 1.1 Create a new Application in your IDP

On Azure Entra ID, search for `Enterprise Application` and click on `New Application`.

<Image img={require('../../img/scim_0.png')}  style={{ width: '800px', height: 'auto' }} />

<br />
<br />

Now select `Create your own application`.

<Image img={require('../../img/scim_1.png')}  style={{ width: '800px', height: 'auto' }} />

<br />
<br />

Enter any name for your application, for example `litellm-scim-app` and select the "Integrate any other application you don't find in the gallery (Non-gallery)" option.

<Image img={require('../../img/scim_3.png')}  style={{ width: '800px', height: 'auto' }} />


## 2. Connect your app on your IDP to LiteLLM SCIM Endpoints

## 2.1 Get your SCIM Tenant URL and Bearer Token

On LiteLLM, navigate to the Settings > Admin Settings > SCIM. On this page you will create a SCIM Token, this allows your IDP to authenticate to litellm `/scim` endpoints.

<Image img={require('../../img/scim_2.png')}  style={{ width: '800px', height: 'auto' }} />

## 2.2 Paste your SCIM Tenant URL and Bearer Token into your IDP Provider


## 3. Test SCIM Connection




