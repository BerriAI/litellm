import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Adding LLM Credentials

You can add LLM provider credentials on the UI. Once you add credentials you can reuse them when adding new models

## Add a credential + model

### 1. Navigate to LLM Credentials page

Go to Models -> LLM Credentials -> Add Credential

<Image img={require('../../img/ui_cred_add.png')} />

### 2. Add credentials

Select your LLM provider, enter your API Key and click "Add Credential"

**Note: Credentials are based on the provider, if you select Vertex AI then you will see `Vertex Project`, `Vertex Location` and `Vertex Credentials` fields**

<Image img={require('../../img/ui_add_cred_2.png')} />


### 3. Use credentials when adding a model

Go to Add Model -> Existing Credentials -> Select your credential in the dropdown

<Image img={require('../../img/ui_cred_3.png')} />


## Create a Credential from an existing model

Use this if you have already created a model and want to store the model credentials for future use

### 1. Select model to create a credential from

Go to Models -> Select your model -> Credential -> Create Credential

<Image img={require('../../img/ui_cred_4.png')} />

### 2. Use new credential when adding a model

Go to Add Model -> Existing Credentials -> Select your credential in the dropdown

<Image img={require('../../img/use_model_cred.png')} />

## Frequently Asked Questions


How are credentials stored?
Credentials in the DB are encrypted/decrypted using `LITELLM_SALT_KEY`, if set. If not, then they are encrypted using `LITELLM_MASTER_KEY`. These keys should be kept secret and not shared with others.


