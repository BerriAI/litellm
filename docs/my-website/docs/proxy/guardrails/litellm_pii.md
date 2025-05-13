import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LiteLLM Native PII Guardrail

LiteLLM provides built-in PII (Personally Identifiable Information) detection and masking capabilities powered by [Presidio](https://github.com/microsoft/presidio).

## Quick Start
### 1. Define Guardrails in your config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "pii-guard"
    litellm_params:
      guardrail: litellm_pii
      mode: "pre_call"
      default_on: true
      entities_config:
        PHONE_NUMBER: "MASK"
        EMAIL_ADDRESS: "MASK"
        PERSON: "MASK"
        LOCATION: "BLOCK"
        CREDIT_CARD: "BLOCK"
```

## Example Usage

<Tabs>
<TabItem label="Masking PII" value="masking">

```yaml
guardrails:
  - guardrail_name: "mask-pii"
    litellm_params:
      guardrail: litellm_pii
      mode: "pre_call"
      entities_config:
        PHONE_NUMBER: "MASK"
        EMAIL_ADDRESS: "MASK"
```

With this configuration, a user message like:
```
My email is john@example.com and my phone is 555-123-4567
```

Will be transformed to:
```
My email is <EMAIL_ADDRESS> and my phone is <PHONE_NUMBER>
```

</TabItem>

<TabItem label="Blocking PII" value="blocking">

```yaml
guardrails:
  - guardrail_name: "block-pii"
    litellm_params:
      guardrail: litellm_pii
      mode: "pre_call"
      entities_config:
        CREDIT_CARD: "BLOCK"
        LOCATION: "BLOCK"
```

If a user sends a message containing a credit card number or location, the request will be blocked entirely, and an error response will be returned.

</TabItem>
</Tabs>

## Modes

- `pre_call`: Run on user input before sending to the LLM
- `post_call`: Run on LLM output before returning to the user
- `during_call`: Run in parallel with the LLM call 



## Supported PII Entity Types

The `entities_config` parameter defines which PII entities to detect and how to handle them:

```yaml
entities_config:
  ENTITY_TYPE: "ACTION"
```

### Available Actions
- `MASK`: Replace detected PII with a placeholder tag (e.g., `<EMAIL_ADDRESS>`)
- `BLOCK`: Block the request entirely if any specified PII is detected


### Available Entity Types



#### Global

|Entity Type | Description | Detection Method |
| --- | --- | --- |
|CREDIT_CARD |A credit card number is between 12 to 19 digits. <https://en.wikipedia.org/wiki/Payment_card_number>|Pattern match and checksum|
|CRYPTO|A Crypto wallet number. Currently only Bitcoin address is supported|Pattern match, context and checksum|
|DATE_TIME|Absolute or relative dates or periods or times smaller than a day.|Pattern match and context|
|EMAIL_ADDRESS|An email address identifies an email box to which email messages are delivered|Pattern match, context and RFC-822 validation|
|IBAN_CODE|The International Bank Account Number (IBAN) is an internationally agreed system of identifying bank accounts across national borders to facilitate the communication and processing of cross border transactions with a reduced risk of transcription errors.|Pattern match, context and checksum|
|IP_ADDRESS|An Internet Protocol (IP) address (either IPv4 or IPv6).|Pattern match, context and checksum|
|NRP|A person’s Nationality, religious or political group.|Custom logic and context|
|LOCATION|Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains|Custom logic and context|
|PERSON|A full person name, which can include first names, middle names or initials, and last names.|Custom logic and context|
|PHONE_NUMBER|A telephone number|Custom logic, pattern match and context|
|MEDICAL_LICENSE|Common medical license numbers.|Pattern match, context and checksum|
|URL|A URL (Uniform Resource Locator), unique identifier used to locate a resource on the Internet|Pattern match, context and top level url validation|

#### USA

|Entity Type|Description|Detection Method|
|--- |--- |--- |
|US_BANK_NUMBER|A US bank account number is between 8 to 17 digits.|Pattern match and context|
|US_DRIVER_LICENSE|A US driver license according to <https://ntsi.com/drivers-license-format/>|Pattern match and context|
|US_ITIN | US Individual Taxpayer Identification Number (ITIN). Nine digits that start with a "9" and contain a "7" or "8" as the 4 digit.|Pattern match and context|
|US_PASSPORT |A US passport number with 9 digits.|Pattern match and context|
|US_SSN|A US Social Security Number (SSN) with 9 digits.|Pattern match and context|

#### UK

|Entity Type|Description|Detection Method|
|--- |--- |--- |
|UK_NHS|A UK NHS number is 10 digits.|Pattern match, context and checksum|
|UK_NINO|UK [National Insurance Number](https://en.wikipedia.org/wiki/National_Insurance_number) is a unique identifier used in the administration of National Insurance and tax.|Pattern match and context|

#### Spain

|Entity Type|Description|Detection Method|
|--- |--- |--- |
|ES_NIF| A spanish NIF number (Personal tax ID) .|Pattern match, context and checksum|
|ES_NIE| A spanish NIE number (Foreigners ID card) .|Pattern match, context and checksum|

#### Italy

|Entity Type|Description|Detection Method|
|--- |--- |--- |
|IT_FISCAL_CODE| An Italian personal identification code. <https://en.wikipedia.org/wiki/Italian_fiscal_code>|Pattern match, context and checksum|
|IT_DRIVER_LICENSE| An Italian driver license number.|Pattern match and context|
|IT_VAT_CODE| An Italian VAT code number |Pattern match, context and checksum|
|IT_PASSPORT|An Italian passport number.|Pattern match and context|
|IT_IDENTITY_CARD|An Italian identity card number. <https://en.wikipedia.org/wiki/Italian_electronic_identity_card>|Pattern match and context|

#### Poland

|Entity Type|Description|Detection Method|
|--- |--- |--- |
|PL_PESEL|Polish PESEL number|Pattern match, context and checksum|

#### Singapore

|FieldType|Description|Detection Method|
|--- |--- |--- |
|SG_NRIC_FIN| A National Registration Identification Card | Pattern match and context |
|SG_UEN| A Unique Entity Number (UEN) is a standard identification number for entities registered in Singapore. | Pattern match, context, and checksum |

#### Australia

|FieldType|Description|Detection Method|
|--- |--- |--- |
|AU_ABN| The Australian Business Number (ABN) is a unique 11 digit identifier issued to all entities registered in the Australian Business Register (ABR). | Pattern match, context, and checksum |
|AU_ACN| An Australian Company Number is a unique nine-digit number issued by the Australian Securities and Investments Commission to every company registered under the Commonwealth Corporations Act 2001 as an identifier. | Pattern match, context, and checksum |
|AU_TFN| The tax file number (TFN) is a unique identifier issued by the Australian Taxation Office to each taxpaying entity | Pattern match, context, and checksum |
|AU_MEDICARE| Medicare number is a unique identifier issued by Australian Government that enables the cardholder to receive a rebates of medical expenses under Australia's Medicare system| Pattern match, context, and checksum |

#### India
| FieldType  | Description                                                                                                                                                         |Detection Method|
|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|--- |
| IN_PAN     | The Indian Permanent Account Number (PAN) is a unique 12 character alphanumeric identifier issued to all business and individual entities registered as Tax Payers. | Pattern match, context |
| IN_AADHAAR | Indian government issued unique 12 digit individual identity number                                                                                                 | Pattern match, context, and checksum |
| IN_VEHICLE_REGISTRATION | Indian government issued transport (govt, personal, diplomatic, defence)  vehicle registration number                                                               | Pattern match, context, and checksum |
| IN_VOTER | Indian Election Commission issued 10 digit alpha numeric voter id for all indian citizens (age 18 or above) | Pattern match, context |
| IN_PASSPORT | Indian Passport Number | Pattern match, Context |

#### Finland
| FieldType  | Description                                                                                             | Detection Method                         |
|------------|---------------------------------------------------------------------------------------------------------|------------------------------------------|
| FI_PERSONAL_IDENTITY_CODE     | The Finnish Personal Identity Code (Henkilötunnus) is a unique 11 character individual identity number. | Pattern match, context and custom logic. |
