import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Onboard Users for AI Exploration

v1.73.0 introduces the ability to assign new users to Default Teams. This makes it much easier to enable experimentation with LLMs within your company, by allowing users to sign in and create $10 keys for AI exploration. 


### 1. Create a team

Create a team called `internal exploration` with:
- `models`:  access to specific models (e.g. `gpt-4o`, `claude-3-5-sonnet`)
- `max budget`: The team max budget will ensure spend for the entire team never exceeds a certain amount. 
- `reset budget`: Set this to monthly. LiteLLM will reset the budget at the start of each month. 
- `team member max budget`: The team member max budget will ensure spend for an individual team member never exceeds a certain amount. 

<Image img={require('../../img/create_default_team.png')}  style={{ width: '600px', height: 'auto' }} />

### 2. Update team member permissions

Click on the team you just created, and update the team member permissions under `Member Permissions`.

This will allow all team members, to create keys. 

<Image img={require('../../img/team_member_permissions.png')}  style={{ width: '600px', height: 'auto' }} />


### 3. Set team as default team

Go to `Internal Users` -> `Default User Settings` and set the default team to the team you just created. 

Let's also set the default models to `no-default-models`. This means a user can only create keys within a team.

<Image img={require('../../img/default_user_settings_with_default_team.png')}  style={{ width: '1000px', height: 'auto' }} />

### 4. Test it! 

Let's create a new user and test it out. 

#### a. Create a new user

Create a new user with email `test_default_team_user@xyz.com`.

<Image img={require('../../img/create_user.png')}  style={{ width: '600px', height: 'auto' }} />

Once you click `Create User`, you will get an invitation link, save it for later. 

#### b. Verify user is added to the team

Click on the created user, and verify they are added to the team. 

We can see the user is added to the team, and has no default models. 

<Image img={require('../../img/user_info_with_default_team.png')}  style={{ width: '1000px', height: 'auto' }} />

#### c. Login as user 

Now use the invitation link from 4a. to login as the user. 

<Image img={require('../../img/new_user_login.png')}  style={{ width: '600px', height: 'auto' }} />

#### d. Verify you can't create keys without specifying a team

You should see a message saying you need to select a team. 

<Image img={require('../../img/create_key_no_team.png')}  style={{ width: '1000px', height: 'auto' }} />

#### e. Verify you can create a key when specifying a team

<Image img={require('../../img/create_key_with_default_team.png')}  style={{ width: '1000px', height: 'auto' }} />

Success! 

You should now see the created key

<Image img={require('../../img/create_key_with_default_team_success.png')}  style={{ width: '600px', height: 'auto' }} />