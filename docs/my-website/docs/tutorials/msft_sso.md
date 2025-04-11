import Image from '@theme/IdealImage';

# Microsoft SSO: Sync Groups, Members with LiteLLM

Sync Microsoft SSO Groups, Members with LiteLLM Teams. 

<Image img={require('../../img/litellm_entra_id.png')}  style={{ width: '800px', height: 'auto' }} />

<br />
<br />

## Overview of this tutorial

1. Auto-Create Entra ID Groups on LiteLLM Teams 
2. Sync Entra ID Team Memberships
3. Set default params for new teams and users auto-created on LiteLLM

## 1. Auto-Create Entra ID Groups on LiteLLM Teams 

### 1.1 Create a new group in Entra ID


Navigate to [your Azure Portal](https://portal.azure.com/) > Groups > New Group. Create a new group. 

<Image img={require('../../img/entra_create_team.png')}  style={{ width: '800px', height: 'auto' }} />

### 1.2 Assign the group to your LiteLLM Enterprise App

On your Azure Portal, navigate to `Enterprise Applications` > Select your litellm app 

<Image img={require('../../img/msft_enterprise_app.png')}  style={{ width: '800px', height: 'auto' }} />

<br />
<br />

Once you've selected your litellm app, click on `Users and Groups` > `Add user/group` 

<Image img={require('../../img/msft_enterprise_assign_group.png')}  style={{ width: '800px', height: 'auto' }} />

<br />

Now select the group you created in step 1.1. And add it to the LiteLLM Enterprise App. At this point we have added `Product LLM Evals Group` to the LiteLLM Enterprise App. The next steps is having LiteLLM automatically create the `Product LLM Evals Group` on the LiteLLM DB when a new user signs in.

<Image img={require('../../img/msft_enterprise_select_group.png')}  style={{ width: '800px', height: 'auto' }} />


### 1.3 Sign in to LiteLLM UI via SSO

### 1.4 Check the new team on LiteLLM UI


## 2. Sync Entra ID Team Memberships

### 2.1 Add a member to the group in Entra ID

### 2.2 Sign in as the new user on LiteLLM UI

### 2.3 Check the team membership on LiteLLM UI

## 3. Set default params for new teams and users auto-created on LiteLLM

### 3.1 Set default params for new teams

### 3.2 Set default params for new users













