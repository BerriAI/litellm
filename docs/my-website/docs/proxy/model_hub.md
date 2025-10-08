import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Model Hub

Tell developers what models are available on the proxy.

This feature is **available in v1.74.3-stable and above**.

## Overview

Admin can select models to expose on public model hub -> Users can go to the public url (`/ui/model_hub_table`) and see available models. 

<Image img={require('../../img/final_public_model_hub_view.png')} />  

## How to use

### 1. Go to the Admin UI

Navigate to the Model Hub page in the Admin UI (`PROXY_BASE_URL/ui/?login=success&page=model-hub-table`)

<Image img={require('../../img/model_hub_admin_view.png')} />  

### 2. Select the models you want to expose

Click on `Make Public` and select the models you want to expose.

<Image img={require('../../img/make_public_modal.png')} />  

### 3. Confirm the changes

<Image img={require('../../img/make_public_modal_confirmation.png')} />  

### 4. Success! 

Go to the public url (`PROXY_BASE_URL/ui/model_hub_table`) and see available models. 

<Image img={require('../../img/final_public_model_hub_view.png')} />  
