(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,191905,e=>{var n=e.i(843476),t=e.i(466828),r=e.i(677572),s=e.i(778917),a=e.i(196631);let i=({href:e,className:t})=>(0,n.jsxs)("a",{href:e,target:"_blank",rel:"noopener noreferrer",title:"Open documentation in a new tab",className:(0,a.c)("inline-flex items-center gap-2 rounded-xl border border-border bg-card/80 px-3.5 py-2 text-sm font-medium text-foreground shadow-xs","hover:bg-card focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring active:translate-y-[0.5px]",t),children:[(0,n.jsx)("span",{children:"API Reference Docs"}),(0,n.jsx)(s.f,{"aria-hidden":!0,className:"h-4 w-4 opacity-80"}),(0,n.jsx)("span",{className:"sr-only",children:"(opens in a new tab)"})]}),o=({proxySettings:e})=>{let s="<your_proxy_base_url>",a=e?.LITELLM_UI_API_DOC_BASE_URL;return a&&a.trim()?s=a:e?.PROXY_BASE_URL&&(s=e.PROXY_BASE_URL),(0,n.jsx)("div",{className:"grid grid-cols-1 gap-2 p-8 h-[80vh] w-full mt-2",children:(0,n.jsxs)("div",{className:"mb-5",children:[(0,n.jsxs)("div",{className:"flex items-center justify-between",children:[(0,n.jsx)("h1",{className:"text-2xl font-semibold text-foreground",children:"OpenAI Compatible Proxy: API Reference"}),(0,n.jsx)(i,{className:"ml-3 shrink-0",href:"https://docs.litellm.ai/docs/proxy/user_keys"})]}),(0,n.jsxs)("p",{className:"mt-2 mb-2 text-sm text-muted-foreground",children:["LiteLLM is OpenAI Compatible. This means your API Key works with the OpenAI SDK. Just replace the base_url to point to your litellm proxy. Example Below"," "]}),(0,n.jsxs)(r.i,{defaultValue:"openai",children:[(0,n.jsxs)(r.U,{variant:"line",className:"border-b rounded-none w-full justify-start h-auto p-0",children:[(0,n.jsx)(r.k,{value:"openai",className:"rounded-none px-4 py-2 flex-none",children:"OpenAI Python SDK"}),(0,n.jsx)(r.k,{value:"llamaindex",className:"rounded-none px-4 py-2 flex-none",children:"LlamaIndex"}),(0,n.jsx)(r.k,{value:"langchain",className:"rounded-none px-4 py-2 flex-none",children:"Langchain Py"})]}),(0,n.jsx)(r.S,{value:"openai",keepMounted:!0,children:(0,n.jsx)(t.f,{language:"python",code:`import openai
client = openai.OpenAI(
    api_key="your_api_key",
    base_url="${s}" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
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

print(response)`})}),(0,n.jsx)(r.S,{value:"llamaindex",keepMounted:!0,children:(0,n.jsx)(t.f,{language:"python",code:`import os, dotenv

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

llm = AzureOpenAI(
    engine="azure-gpt-3.5",               # model_name on litellm proxy
    temperature=0.0,
    azure_endpoint="${s}", # litellm proxy endpoint
    api_key="<your-master-key>",          # litellm proxy API Key
    api_version="2023-07-01-preview",
)

embed_model = AzureOpenAIEmbedding(
    deployment_name="azure-embedding-model",
    azure_endpoint="${s}",
    api_key="<your-master-key>",
    api_version="2023-07-01-preview",
)

documents = SimpleDirectoryReader("llama_index_data").load_data()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)`})}),(0,n.jsx)(r.S,{value:"langchain",keepMounted:!0,children:(0,n.jsx)(t.f,{language:"python",code:`from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="${s}",
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

print(response)`})})]})]})})};var l=e.i(541202),m=e.i(135214),d=e.i(592392);e.s(["default",0,()=>{let{accessToken:e}=(0,m.f)(),t=(0,d.U)(e);return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(l.f,{featureName:"The API Reference tab"}),(0,n.jsx)(o,{proxySettings:t})]})}],191905)},541202,e=>{var n=e.i(843476),t=e.i(522016),r=e.i(952571),s=e.i(929592);e.s(["f",0,({featureName:e})=>(0,n.jsxs)(s.J,{className:"mb-4",children:[(0,n.jsx)(r.f,{}),(0,n.jsx)(s.s,{children:`${e} is deprecated`}),(0,n.jsxs)(s.p,{children:[`${e} will be removed in the near future. See the `,(0,n.jsx)(t.default,{href:"https://github.com/BerriAI/litellm/discussions/32090",target:"_blank",rel:"noopener noreferrer",children:"deprecation discussion"})," for details."]})]})])}])})();