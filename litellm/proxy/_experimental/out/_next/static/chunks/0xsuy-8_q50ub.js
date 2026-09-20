(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,191905,e=>{"use strict";var t=e.i(843476),n=e.i(466828),s=e.i(677572),r=e.i(778917),a=e.i(196631);let o=({href:e,className:n})=>(0,t.jsxs)("a",{href:e,target:"_blank",rel:"noopener noreferrer",title:"Open documentation in a new tab",className:(0,a.cn)("inline-flex items-center gap-2 rounded-xl border border-border bg-card/80 px-3.5 py-2 text-sm font-medium text-foreground shadow-xs","hover:bg-card focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring active:translate-y-[0.5px]",n),children:[(0,t.jsx)("span",{children:"API Reference Docs"}),(0,t.jsx)(r.ExternalLink,{"aria-hidden":!0,className:"h-4 w-4 opacity-80"}),(0,t.jsx)("span",{className:"sr-only",children:"(opens in a new tab)"})]}),i=({proxySettings:e})=>{let r="<your_proxy_base_url>",a=e?.LITELLM_UI_API_DOC_BASE_URL;return a&&a.trim()?r=a:e?.PROXY_BASE_URL&&(r=e.PROXY_BASE_URL),(0,t.jsx)("div",{className:"grid grid-cols-1 gap-2 p-8 h-[80vh] w-full mt-2",children:(0,t.jsxs)("div",{className:"mb-5",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between",children:[(0,t.jsx)("h1",{className:"text-2xl font-semibold text-foreground",children:"OpenAI Compatible Proxy: API Reference"}),(0,t.jsx)(o,{className:"ml-3 shrink-0",href:"https://docs.litellm.ai/docs/proxy/user_keys"})]}),(0,t.jsxs)("p",{className:"mt-2 mb-2 text-sm text-muted-foreground",children:["LiteLLM is OpenAI Compatible. This means your API Key works with the OpenAI SDK. Just replace the base_url to point to your litellm proxy. Example Below"," "]}),(0,t.jsxs)(s.Tabs,{defaultValue:"openai",children:[(0,t.jsxs)(s.TabsList,{variant:"line",className:"border-b rounded-none w-full justify-start h-auto p-0",children:[(0,t.jsx)(s.TabsTrigger,{value:"openai",className:"rounded-none px-4 py-2 flex-none",children:"OpenAI Python SDK"}),(0,t.jsx)(s.TabsTrigger,{value:"llamaindex",className:"rounded-none px-4 py-2 flex-none",children:"LlamaIndex"}),(0,t.jsx)(s.TabsTrigger,{value:"langchain",className:"rounded-none px-4 py-2 flex-none",children:"Langchain Py"})]}),(0,t.jsx)(s.TabsContent,{value:"openai",keepMounted:!0,children:(0,t.jsx)(n.default,{language:"python",code:`import openai
client = openai.OpenAI(
    api_key="your_api_key",
    base_url="${r}" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)`})}),(0,t.jsx)(s.TabsContent,{value:"llamaindex",keepMounted:!0,children:(0,t.jsx)(n.default,{language:"python",code:`import os, dotenv

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

llm = AzureOpenAI(
    engine="azure-gpt-3.5",               # model_name on litellm proxy
    temperature=0.0,
    azure_endpoint="${r}", # litellm proxy endpoint
    api_key="sk-1234",                    # litellm proxy API Key
    api_version="2023-07-01-preview",
)

embed_model = AzureOpenAIEmbedding(
    deployment_name="azure-embedding-model",
    azure_endpoint="${r}",
    api_key="sk-1234",
    api_version="2023-07-01-preview",
)

documents = SimpleDirectoryReader("llama_index_data").load_data()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)`})}),(0,t.jsx)(s.TabsContent,{value:"langchain",keepMounted:!0,children:(0,t.jsx)(n.default,{language:"python",code:`from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="${r}",
    model = "gpt-3.5-turbo",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)`})})]})]})})};var l=e.i(541202),d=e.i(135214),m=e.i(592392);e.s(["default",0,()=>{let{accessToken:e}=(0,d.default)(),n=(0,m.default)(e);return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(l.DeprecationBanner,{featureName:"The API Reference tab"}),(0,t.jsx)(i,{proxySettings:n})]})}],191905)},541202,e=>{"use strict";var t=e.i(843476),n=e.i(271645),s=e.i(522016),r=e.i(952571),a=e.i(37727);e.s(["DeprecationBanner",0,({featureName:e})=>{let[o,i]=(0,n.useState)(!1);return o?null:(0,t.jsxs)("div",{role:"alert",className:"mb-4 flex items-start gap-3 rounded-lg border border-border bg-muted/50 px-4 py-3 text-sm",children:[(0,t.jsx)(r.Info,{className:"mt-0.5 size-4 shrink-0 text-muted-foreground"}),(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("p",{className:"font-medium",children:`${e} is on a draft deprecation list`}),(0,t.jsxs)("p",{className:"mt-1 break-words text-muted-foreground",children:[`${e} is one of several experimental features we're considering removing, potentially as early as September 1, 2026. This list is a draft and is not final. If you rely on this feature, please share feedback on the `,(0,t.jsx)(s.default,{href:"https://github.com/BerriAI/litellm/discussions/32090",target:"_blank",rel:"noopener noreferrer",className:"underline underline-offset-4",children:"deprecation discussion"}),"."]})]}),(0,t.jsx)("button",{type:"button","aria-label":"Close",onClick:()=>i(!0),className:"shrink-0 rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground",children:(0,t.jsx)(a.X,{className:"size-4"})})]})}])}]);