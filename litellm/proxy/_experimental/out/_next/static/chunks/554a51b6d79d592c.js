(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,916925,e=>{"use strict";var t,a=((t={}).A2A_Agent="A2A Agent",t.AI21="Ai21",t.AI21_CHAT="Ai21 Chat",t.AIML="AI/ML API",t.AIOHTTP_OPENAI="Aiohttp Openai",t.Anthropic="Anthropic",t.ANTHROPIC_TEXT="Anthropic Text",t.AssemblyAI="AssemblyAI",t.AUTO_ROUTER="Auto Router",t.Bedrock="Amazon Bedrock",t.BedrockMantle="Amazon Bedrock Mantle",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.AZURE_TEXT="Azure Text",t.BASETEN="Baseten",t.BYTEZ="Bytez",t.Cerebras="Cerebras",t.CLARIFAI="Clarifai",t.CLOUDFLARE="Cloudflare",t.CODESTRAL="Codestral",t.Cohere="Cohere",t.COHERE_CHAT="Cohere Chat",t.COMETAPI="Cometapi",t.COMPACTIFAI="Compactifai",t.Cursor="Cursor",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DATAROBOT="Datarobot",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.DOCKER_MODEL_RUNNER="Docker Model Runner",t.DOTPROMPT="Dotprompt",t.ElevenLabs="ElevenLabs",t.EMPOWER="Empower",t.FalAI="Fal AI",t.FEATHERLESS_AI="Featherless Ai",t.FireworksAI="Fireworks AI",t.FRIENDLIAI="Friendliai",t.GALADRIEL="Galadriel",t.GITHUB_COPILOT="Github Copilot",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.HEROKU="Heroku",t.Hosted_Vllm="vllm",t.HUGGINGFACE="Huggingface",t.HYPERBOLIC="Hyperbolic",t.Infinity="Infinity",t.JinaAI="Jina AI",t.LAMBDA_AI="Lambda Ai",t.LEMONADE="Lemonade",t.LLAMAFILE="Llamafile",t.LM_STUDIO="Lm Studio",t.LLAMA="Meta Llama",t.MARITALK="Maritalk",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.MOONSHOT="Moonshot",t.MORPH="Morph",t.NEBIUS="Nebius",t.NLP_CLOUD="Nlp Cloud",t.NOVITA="Novita",t.NSCALE="Nscale",t.NVIDIA_NIM="Nvidia Nim",t.Ollama="Ollama",t.OLLAMA_CHAT="Ollama Chat",t.OOBABOOGA="Oobabooga",t.OpenAI="OpenAI",t.OPENAI_LIKE="Openai Like",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.OVHCLOUD="Ovhcloud",t.Perplexity="Perplexity",t.PETALS="Petals",t.PG_VECTOR="Pg Vector",t.PREDIBASE="Predibase",t.RECRAFT="Recraft",t.REPLICATE="Replicate",t.RunwayML="RunwayML",t.SAGEMAKER_LEGACY="Sagemaker",t.Sambanova="Sambanova",t.SAP="SAP Generative AI Hub",t.Snowflake="Snowflake",t.TEXT_COMPLETION_CODESTRAL="Text-Completion-Codestral",t.TogetherAI="TogetherAI",t.TOPAZ="Topaz",t.Triton="Triton",t.V0="V0",t.VERCEL_AI_GATEWAY="Vercel Ai Gateway",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VERTEX_AI_BETA="Vertex Ai Beta",t.VLLM="Vllm",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.WANDB="Wandb",t.WATSONX="Watsonx",t.WATSONX_TEXT="Watsonx Text",t.xAI="xAI",t.XINFERENCE="Xinference",t.ZAI="Z.AI (Zhipu AI)",t);let i={A2A_Agent:"a2a_agent",AI21:"ai21",AI21_CHAT:"ai21_chat",AIML:"aiml",AIOHTTP_OPENAI:"aiohttp_openai",Anthropic:"anthropic",ANTHROPIC_TEXT:"anthropic_text",AssemblyAI:"assemblyai",AUTO_ROUTER:"auto_router",Azure:"azure",Azure_AI_Studio:"azure_ai",AZURE_TEXT:"azure_text",BASETEN:"baseten",Bedrock:"bedrock",BedrockMantle:"bedrock_mantle",BYTEZ:"bytez",Cerebras:"cerebras",CLARIFAI:"clarifai",CLOUDFLARE:"cloudflare",CODESTRAL:"codestral",Cohere:"cohere",COHERE_CHAT:"cohere_chat",COMETAPI:"cometapi",COMPACTIFAI:"compactifai",Cursor:"cursor",Dashscope:"dashscope",Databricks:"databricks",DATAROBOT:"datarobot",DeepInfra:"deepinfra",Deepgram:"deepgram",Deepseek:"deepseek",DOCKER_MODEL_RUNNER:"docker_model_runner",DOTPROMPT:"dotprompt",ElevenLabs:"elevenlabs",EMPOWER:"empower",FalAI:"fal_ai",FEATHERLESS_AI:"featherless_ai",FireworksAI:"fireworks_ai",FRIENDLIAI:"friendliai",GALADRIEL:"galadriel",GITHUB_COPILOT:"github_copilot",Google_AI_Studio:"gemini",GradientAI:"gradient_ai",Groq:"groq",HEROKU:"heroku",Hosted_Vllm:"hosted_vllm",HUGGINGFACE:"huggingface",HYPERBOLIC:"hyperbolic",Infinity:"infinity",JinaAI:"jina_ai",LAMBDA_AI:"lambda_ai",LEMONADE:"lemonade",LLAMAFILE:"llamafile",LLAMA:"meta_llama",LM_STUDIO:"lm_studio",MARITALK:"maritalk",MiniMax:"minimax",MistralAI:"mistral",MOONSHOT:"moonshot",MORPH:"morph",NEBIUS:"nebius",NLP_CLOUD:"nlp_cloud",NOVITA:"novita",NSCALE:"nscale",NVIDIA_NIM:"nvidia_nim",Ollama:"ollama",OLLAMA_CHAT:"ollama_chat",OOBABOOGA:"oobabooga",OpenAI:"openai",OPENAI_LIKE:"openai_like",OpenAI_Compatible:"openai",OpenAI_Text:"text-completion-openai",OpenAI_Text_Compatible:"text-completion-openai",Openrouter:"openrouter",Oracle:"oci",OVHCLOUD:"ovhcloud",Perplexity:"perplexity",PETALS:"petals",PG_VECTOR:"pg_vector",PREDIBASE:"predibase",RECRAFT:"recraft",REPLICATE:"replicate",RunwayML:"runwayml",SAGEMAKER_LEGACY:"sagemaker",SageMaker:"sagemaker_chat",Sambanova:"sambanova",SAP:"sap",Snowflake:"snowflake",TEXT_COMPLETION_CODESTRAL:"text-completion-codestral",TogetherAI:"together_ai",TOPAZ:"topaz",Triton:"triton",V0:"v0",VERCEL_AI_GATEWAY:"vercel_ai_gateway",Vertex_AI:"vertex_ai",VERTEX_AI_BETA:"vertex_ai_beta",VLLM:"vllm",VolcEngine:"volcengine",Voyage:"voyage",WANDB:"wandb",WATSONX:"watsonx",WATSONX_TEXT:"watsonx_text",xAI:"xai",XINFERENCE:"xinference",ZAI:"zai"},r="../ui/assets/logos/",n={"A2A Agent":`${r}a2a_agent.png`,Ai21:`${r}ai21.svg`,"Ai21 Chat":`${r}ai21.svg`,"AI/ML API":`${r}aiml_api.svg`,"Aiohttp Openai":`${r}openai_small.svg`,Anthropic:`${r}anthropic.svg`,"Anthropic Text":`${r}anthropic.svg`,AssemblyAI:`${r}assemblyai_small.png`,Azure:`${r}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${r}microsoft_azure.svg`,"Azure Text":`${r}microsoft_azure.svg`,Baseten:`${r}baseten.svg`,"Amazon Bedrock":`${r}bedrock.svg`,"Amazon Bedrock Mantle":`${r}bedrock.svg`,"AWS SageMaker":`${r}bedrock.svg`,Cerebras:`${r}cerebras.svg`,Cloudflare:`${r}cloudflare.svg`,Codestral:`${r}mistral.svg`,Cohere:`${r}cohere.svg`,"Cohere Chat":`${r}cohere.svg`,Cometapi:`${r}cometapi.svg`,Cursor:`${r}cursor.svg`,"Databricks (Qwen API)":`${r}databricks.svg`,Dashscope:`${r}dashscope.svg`,Deepseek:`${r}deepseek.svg`,Deepgram:`${r}deepgram.png`,DeepInfra:`${r}deepinfra.png`,ElevenLabs:`${r}elevenlabs.png`,"Fal AI":`${r}fal_ai.jpg`,"Featherless Ai":`${r}featherless.svg`,"Fireworks AI":`${r}fireworks.svg`,Friendliai:`${r}friendli.svg`,"Github Copilot":`${r}github_copilot.svg`,"Google AI Studio":`${r}google.svg`,GradientAI:`${r}gradientai.svg`,Groq:`${r}groq.svg`,vllm:`${r}vllm.png`,Huggingface:`${r}huggingface.svg`,Hyperbolic:`${r}hyperbolic.svg`,Infinity:`${r}infinity.png`,"Jina AI":`${r}jina.png`,"Lambda Ai":`${r}lambda.svg`,"Lm Studio":`${r}lmstudio.svg`,"Meta Llama":`${r}meta_llama.svg`,MiniMax:`${r}minimax.svg`,"Mistral AI":`${r}mistral.svg`,Moonshot:`${r}moonshot.svg`,Morph:`${r}morph.svg`,Nebius:`${r}nebius.svg`,Novita:`${r}novita.svg`,"Nvidia Nim":`${r}nvidia_nim.svg`,Ollama:`${r}ollama.svg`,"Ollama Chat":`${r}ollama.svg`,Oobabooga:`${r}openai_small.svg`,OpenAI:`${r}openai_small.svg`,"Openai Like":`${r}openai_small.svg`,"OpenAI Text Completion":`${r}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${r}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${r}openai_small.svg`,Openrouter:`${r}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${r}oracle.svg`,Perplexity:`${r}perplexity-ai.svg`,Recraft:`${r}recraft.svg`,Replicate:`${r}replicate.svg`,RunwayML:`${r}runwayml.png`,Sagemaker:`${r}bedrock.svg`,Sambanova:`${r}sambanova.svg`,"SAP Generative AI Hub":`${r}sap.png`,Snowflake:`${r}snowflake.svg`,"Text-Completion-Codestral":`${r}mistral.svg`,TogetherAI:`${r}togetherai.svg`,Topaz:`${r}topaz.svg`,Triton:`${r}nvidia_triton.png`,V0:`${r}v0.svg`,"Vercel Ai Gateway":`${r}vercel.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${r}google.svg`,"Vertex Ai Beta":`${r}google.svg`,Vllm:`${r}vllm.png`,VolcEngine:`${r}volcengine.png`,"Voyage AI":`${r}voyage.webp`,Watsonx:`${r}watsonx.svg`,"Watsonx Text":`${r}watsonx.svg`,xAI:`${r}xai.svg`,Xinference:`${r}xinference.svg`};e.s(["Providers",()=>a,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else if("Cursor"===e)return"cursor/claude-4-sonnet";else if("Z.AI (Zhipu AI)"===e)return"zai/glm-4.5";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:n[e],displayName:e}}let t=Object.keys(i).find(t=>i[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let r=a[t];return{logo:n[r],displayName:r}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let a=i[e];console.log(`Provider mapped to: ${a}`);let r=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let i=t.litellm_provider;(i===a||"string"==typeof i&&(i.startsWith(`${a}_`)||i.startsWith(`${a}-`)))&&r.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&r.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&r.push(e)}))),r},"providerLogoMap",0,n,"provider_map",0,i])},185793,e=>{"use strict";e.i(247167);var t=e.i(271645),a=e.i(343794),i=e.i(242064),r=e.i(529681);let n=e=>{let{prefixCls:i,className:r,style:n,size:o,shape:l}=e,s=(0,a.default)({[`${i}-lg`]:"large"===o,[`${i}-sm`]:"small"===o}),p=(0,a.default)({[`${i}-circle`]:"circle"===l,[`${i}-square`]:"square"===l,[`${i}-round`]:"round"===l}),c=t.useMemo(()=>"number"==typeof o?{width:o,height:o,lineHeight:`${o}px`}:{},[o]);return t.createElement("span",{className:(0,a.default)(i,s,p,r),style:Object.assign(Object.assign({},c),n)})};e.i(296059);var o=e.i(694758),l=e.i(915654),s=e.i(246422),p=e.i(838378);let c=new o.Keyframes("ant-skeleton-loading",{"0%":{backgroundPosition:"100% 50%"},"100%":{backgroundPosition:"0 50%"}}),g=e=>({height:e,lineHeight:(0,l.unit)(e)}),u=e=>Object.assign({width:e},g(e)),d=(e,t)=>Object.assign({width:t(e).mul(5).equal(),minWidth:t(e).mul(5).equal()},g(e)),m=e=>Object.assign({width:e},g(e)),f=(e,t,a)=>{let{skeletonButtonCls:i}=e;return{[`${a}${i}-circle`]:{width:t,minWidth:t,borderRadius:"50%"},[`${a}${i}-round`]:{borderRadius:t}}},_=(e,t)=>Object.assign({width:t(e).mul(2).equal(),minWidth:t(e).mul(2).equal()},g(e)),h=(0,s.genStyleHooks)("Skeleton",e=>{let{componentCls:t,calc:a}=e;return(e=>{let{componentCls:t,skeletonAvatarCls:a,skeletonTitleCls:i,skeletonParagraphCls:r,skeletonButtonCls:n,skeletonInputCls:o,skeletonImageCls:l,controlHeight:s,controlHeightLG:p,controlHeightSM:g,gradientFromColor:h,padding:b,marginSM:A,borderRadius:v,titleHeight:O,blockRadius:I,paragraphLiHeight:$,controlHeightXS:E,paragraphMarginTop:C}=e;return{[t]:{display:"table",width:"100%",[`${t}-header`]:{display:"table-cell",paddingInlineEnd:b,verticalAlign:"top",[a]:Object.assign({display:"inline-block",verticalAlign:"top",background:h},u(s)),[`${a}-circle`]:{borderRadius:"50%"},[`${a}-lg`]:Object.assign({},u(p)),[`${a}-sm`]:Object.assign({},u(g))},[`${t}-content`]:{display:"table-cell",width:"100%",verticalAlign:"top",[i]:{width:"100%",height:O,background:h,borderRadius:I,[`+ ${r}`]:{marginBlockStart:g}},[r]:{padding:0,"> li":{width:"100%",height:$,listStyle:"none",background:h,borderRadius:I,"+ li":{marginBlockStart:E}}},[`${r}> li:last-child:not(:first-child):not(:nth-child(2))`]:{width:"61%"}},[`&-round ${t}-content`]:{[`${i}, ${r} > li`]:{borderRadius:v}}},[`${t}-with-avatar ${t}-content`]:{[i]:{marginBlockStart:A,[`+ ${r}`]:{marginBlockStart:C}}},[`${t}${t}-element`]:Object.assign(Object.assign(Object.assign(Object.assign({display:"inline-block",width:"auto"},(e=>{let{borderRadiusSM:t,skeletonButtonCls:a,controlHeight:i,controlHeightLG:r,controlHeightSM:n,gradientFromColor:o,calc:l}=e;return Object.assign(Object.assign(Object.assign(Object.assign(Object.assign({[a]:Object.assign({display:"inline-block",verticalAlign:"top",background:o,borderRadius:t,width:l(i).mul(2).equal(),minWidth:l(i).mul(2).equal()},_(i,l))},f(e,i,a)),{[`${a}-lg`]:Object.assign({},_(r,l))}),f(e,r,`${a}-lg`)),{[`${a}-sm`]:Object.assign({},_(n,l))}),f(e,n,`${a}-sm`))})(e)),(e=>{let{skeletonAvatarCls:t,gradientFromColor:a,controlHeight:i,controlHeightLG:r,controlHeightSM:n}=e;return{[t]:Object.assign({display:"inline-block",verticalAlign:"top",background:a},u(i)),[`${t}${t}-circle`]:{borderRadius:"50%"},[`${t}${t}-lg`]:Object.assign({},u(r)),[`${t}${t}-sm`]:Object.assign({},u(n))}})(e)),(e=>{let{controlHeight:t,borderRadiusSM:a,skeletonInputCls:i,controlHeightLG:r,controlHeightSM:n,gradientFromColor:o,calc:l}=e;return{[i]:Object.assign({display:"inline-block",verticalAlign:"top",background:o,borderRadius:a},d(t,l)),[`${i}-lg`]:Object.assign({},d(r,l)),[`${i}-sm`]:Object.assign({},d(n,l))}})(e)),(e=>{let{skeletonImageCls:t,imageSizeBase:a,gradientFromColor:i,borderRadiusSM:r,calc:n}=e;return{[t]:Object.assign(Object.assign({display:"inline-flex",alignItems:"center",justifyContent:"center",verticalAlign:"middle",background:i,borderRadius:r},m(n(a).mul(2).equal())),{[`${t}-path`]:{fill:"#bfbfbf"},[`${t}-svg`]:Object.assign(Object.assign({},m(a)),{maxWidth:n(a).mul(4).equal(),maxHeight:n(a).mul(4).equal()}),[`${t}-svg${t}-svg-circle`]:{borderRadius:"50%"}}),[`${t}${t}-circle`]:{borderRadius:"50%"}}})(e)),[`${t}${t}-block`]:{width:"100%",[n]:{width:"100%"},[o]:{width:"100%"}},[`${t}${t}-active`]:{[`
        ${i},
        ${r} > li,
        ${a},
        ${n},
        ${o},
        ${l}
      `]:Object.assign({},{background:e.skeletonLoadingBackground,backgroundSize:"400% 100%",animationName:c,animationDuration:e.skeletonLoadingMotionDuration,animationTimingFunction:"ease",animationIterationCount:"infinite"})}}})((0,p.mergeToken)(e,{skeletonAvatarCls:`${t}-avatar`,skeletonTitleCls:`${t}-title`,skeletonParagraphCls:`${t}-paragraph`,skeletonButtonCls:`${t}-button`,skeletonInputCls:`${t}-input`,skeletonImageCls:`${t}-image`,imageSizeBase:a(e.controlHeight).mul(1.5).equal(),borderRadius:100,skeletonLoadingBackground:`linear-gradient(90deg, ${e.gradientFromColor} 25%, ${e.gradientToColor} 37%, ${e.gradientFromColor} 63%)`,skeletonLoadingMotionDuration:"1.4s"}))},e=>{let{colorFillContent:t,colorFill:a}=e;return{color:t,colorGradientEnd:a,gradientFromColor:t,gradientToColor:a,titleHeight:e.controlHeight/2,blockRadius:e.borderRadiusSM,paragraphMarginTop:e.marginLG+e.marginXXS,paragraphLiHeight:e.controlHeight/2}},{deprecatedTokens:[["color","gradientFromColor"],["colorGradientEnd","gradientToColor"]]}),b=e=>{let{prefixCls:i,className:r,style:n,rows:o=0}=e,l=Array.from({length:o}).map((a,i)=>t.createElement("li",{key:i,style:{width:((e,t)=>{let{width:a,rows:i=2}=t;return Array.isArray(a)?a[e]:i-1===e?a:void 0})(i,e)}}));return t.createElement("ul",{className:(0,a.default)(i,r),style:n},l)},A=({prefixCls:e,className:i,width:r,style:n})=>t.createElement("h3",{className:(0,a.default)(e,i),style:Object.assign({width:r},n)});function v(e){return e&&"object"==typeof e?e:{}}let O=e=>{let{prefixCls:r,loading:o,className:l,rootClassName:s,style:p,children:c,avatar:g=!1,title:u=!0,paragraph:d=!0,active:m,round:f}=e,{getPrefixCls:_,direction:O,className:I,style:$}=(0,i.useComponentConfig)("skeleton"),E=_("skeleton",r),[C,y,x]=h(E);if(o||!("loading"in e)){let e,i,r=!!g,o=!!u,c=!!d;if(r){let a=Object.assign(Object.assign({prefixCls:`${E}-avatar`},o&&!c?{size:"large",shape:"square"}:{size:"large",shape:"circle"}),v(g));e=t.createElement("div",{className:`${E}-header`},t.createElement(n,Object.assign({},a)))}if(o||c){let e,a;if(o){let a=Object.assign(Object.assign({prefixCls:`${E}-title`},!r&&c?{width:"38%"}:r&&c?{width:"50%"}:{}),v(u));e=t.createElement(A,Object.assign({},a))}if(c){let e,i=Object.assign(Object.assign({prefixCls:`${E}-paragraph`},(e={},r&&o||(e.width="61%"),!r&&o?e.rows=3:e.rows=2,e)),v(d));a=t.createElement(b,Object.assign({},i))}i=t.createElement("div",{className:`${E}-content`},e,a)}let _=(0,a.default)(E,{[`${E}-with-avatar`]:r,[`${E}-active`]:m,[`${E}-rtl`]:"rtl"===O,[`${E}-round`]:f},I,l,s,y,x);return C(t.createElement("div",{className:_,style:Object.assign(Object.assign({},$),p)},e,i))}return null!=c?c:null};O.Button=e=>{let{prefixCls:o,className:l,rootClassName:s,active:p,block:c=!1,size:g="default"}=e,{getPrefixCls:u}=t.useContext(i.ConfigContext),d=u("skeleton",o),[m,f,_]=h(d),b=(0,r.default)(e,["prefixCls"]),A=(0,a.default)(d,`${d}-element`,{[`${d}-active`]:p,[`${d}-block`]:c},l,s,f,_);return m(t.createElement("div",{className:A},t.createElement(n,Object.assign({prefixCls:`${d}-button`,size:g},b))))},O.Avatar=e=>{let{prefixCls:o,className:l,rootClassName:s,active:p,shape:c="circle",size:g="default"}=e,{getPrefixCls:u}=t.useContext(i.ConfigContext),d=u("skeleton",o),[m,f,_]=h(d),b=(0,r.default)(e,["prefixCls","className"]),A=(0,a.default)(d,`${d}-element`,{[`${d}-active`]:p},l,s,f,_);return m(t.createElement("div",{className:A},t.createElement(n,Object.assign({prefixCls:`${d}-avatar`,shape:c,size:g},b))))},O.Input=e=>{let{prefixCls:o,className:l,rootClassName:s,active:p,block:c,size:g="default"}=e,{getPrefixCls:u}=t.useContext(i.ConfigContext),d=u("skeleton",o),[m,f,_]=h(d),b=(0,r.default)(e,["prefixCls"]),A=(0,a.default)(d,`${d}-element`,{[`${d}-active`]:p,[`${d}-block`]:c},l,s,f,_);return m(t.createElement("div",{className:A},t.createElement(n,Object.assign({prefixCls:`${d}-input`,size:g},b))))},O.Image=e=>{let{prefixCls:r,className:n,rootClassName:o,style:l,active:s}=e,{getPrefixCls:p}=t.useContext(i.ConfigContext),c=p("skeleton",r),[g,u,d]=h(c),m=(0,a.default)(c,`${c}-element`,{[`${c}-active`]:s},n,o,u,d);return g(t.createElement("div",{className:m},t.createElement("div",{className:(0,a.default)(`${c}-image`,n),style:l},t.createElement("svg",{viewBox:"0 0 1098 1024",xmlns:"http://www.w3.org/2000/svg",className:`${c}-image-svg`},t.createElement("title",null,"Image placeholder"),t.createElement("path",{d:"M365.714286 329.142857q0 45.714286-32.036571 77.677714t-77.677714 32.036571-77.677714-32.036571-32.036571-77.677714 32.036571-77.677714 77.677714-32.036571 77.677714 32.036571 32.036571 77.677714zM950.857143 548.571429l0 256-804.571429 0 0-109.714286 182.857143-182.857143 91.428571 91.428571 292.571429-292.571429zM1005.714286 146.285714l-914.285714 0q-7.460571 0-12.873143 5.412571t-5.412571 12.873143l0 694.857143q0 7.460571 5.412571 12.873143t12.873143 5.412571l914.285714 0q7.460571 0 12.873143-5.412571t5.412571-12.873143l0-694.857143q0-7.460571-5.412571-12.873143t-12.873143-5.412571zM1097.142857 164.571429l0 694.857143q0 37.741714-26.843429 64.585143t-64.585143 26.843429l-914.285714 0q-37.741714 0-64.585143-26.843429t-26.843429-64.585143l0-694.857143q0-37.741714 26.843429-64.585143t64.585143-26.843429l914.285714 0q37.741714 0 64.585143 26.843429t26.843429 64.585143z",className:`${c}-image-path`})))))},O.Node=e=>{let{prefixCls:r,className:n,rootClassName:o,style:l,active:s,children:p}=e,{getPrefixCls:c}=t.useContext(i.ConfigContext),g=c("skeleton",r),[u,d,m]=h(g),f=(0,a.default)(g,`${g}-element`,{[`${g}-active`]:s},d,n,o,m);return u(t.createElement("div",{className:f},t.createElement("div",{className:(0,a.default)(`${g}-image`,n),style:l},p)))},e.s(["default",0,O],185793)},959013,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M482 152h60q8 0 8 8v704q0 8-8 8h-60q-8 0-8-8V160q0-8 8-8z"}},{tag:"path",attrs:{d:"M192 474h672q8 0 8 8v60q0 8-8 8H160q-8 0-8-8v-60q0-8 8-8z"}}]},name:"plus",theme:"outlined"};var r=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(r.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["default",0,n],959013)},282786,836938,310730,829672,e=>{"use strict";e.i(247167);var t=e.i(271645),a=e.i(343794),i=e.i(914949),r=e.i(404948);let n=e=>e?"function"==typeof e?e():e:null;e.s(["getRenderPropValue",0,n],836938);var o=e.i(613541),l=e.i(763731),s=e.i(242064),p=e.i(491816);e.i(793154);var c=e.i(880476),g=e.i(183293),u=e.i(717356),d=e.i(320560),m=e.i(307358),f=e.i(246422),_=e.i(838378),h=e.i(617933);let b=(0,f.genStyleHooks)("Popover",e=>{let{colorBgElevated:t,colorText:a}=e,i=(0,_.mergeToken)(e,{popoverBg:t,popoverColor:a});return[(e=>{let{componentCls:t,popoverColor:a,titleMinWidth:i,fontWeightStrong:r,innerPadding:n,boxShadowSecondary:o,colorTextHeading:l,borderRadiusLG:s,zIndexPopup:p,titleMarginBottom:c,colorBgElevated:u,popoverBg:m,titleBorderBottom:f,innerContentPadding:_,titlePadding:h}=e;return[{[t]:Object.assign(Object.assign({},(0,g.resetComponent)(e)),{position:"absolute",top:0,left:{_skip_check_:!0,value:0},zIndex:p,fontWeight:"normal",whiteSpace:"normal",textAlign:"start",cursor:"auto",userSelect:"text","--valid-offset-x":"var(--arrow-offset-horizontal, var(--arrow-x))",transformOrigin:"var(--valid-offset-x, 50%) var(--arrow-y, 50%)","--antd-arrow-background-color":u,width:"max-content",maxWidth:"100vw","&-rtl":{direction:"rtl"},"&-hidden":{display:"none"},[`${t}-content`]:{position:"relative"},[`${t}-inner`]:{backgroundColor:m,backgroundClip:"padding-box",borderRadius:s,boxShadow:o,padding:n},[`${t}-title`]:{minWidth:i,marginBottom:c,color:l,fontWeight:r,borderBottom:f,padding:h},[`${t}-inner-content`]:{color:a,padding:_}})},(0,d.default)(e,"var(--antd-arrow-background-color)"),{[`${t}-pure`]:{position:"relative",maxWidth:"none",margin:e.sizePopupArrow,display:"inline-block",[`${t}-content`]:{display:"inline-block"}}}]})(i),(e=>{let{componentCls:t}=e;return{[t]:h.PresetColors.map(a=>{let i=e[`${a}6`];return{[`&${t}-${a}`]:{"--antd-arrow-background-color":i,[`${t}-inner`]:{backgroundColor:i},[`${t}-arrow`]:{background:"transparent"}}}})}})(i),(0,u.initZoomMotion)(i,"zoom-big")]},e=>{let{lineWidth:t,controlHeight:a,fontHeight:i,padding:r,wireframe:n,zIndexPopupBase:o,borderRadiusLG:l,marginXS:s,lineType:p,colorSplit:c,paddingSM:g}=e,u=a-i;return Object.assign(Object.assign(Object.assign({titleMinWidth:177,zIndexPopup:o+30},(0,m.getArrowToken)(e)),(0,d.getArrowOffsetToken)({contentRadius:l,limitVerticalRadius:!0})),{innerPadding:12*!n,titleMarginBottom:n?0:s,titlePadding:n?`${u/2}px ${r}px ${u/2-t}px`:0,titleBorderBottom:n?`${t}px ${p} ${c}`:"none",innerContentPadding:n?`${g}px ${r}px`:0})},{resetStyle:!1,deprecatedTokens:[["width","titleMinWidth"],["minWidth","titleMinWidth"]]});var A=function(e,t){var a={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(a[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,i=Object.getOwnPropertySymbols(e);r<i.length;r++)0>t.indexOf(i[r])&&Object.prototype.propertyIsEnumerable.call(e,i[r])&&(a[i[r]]=e[i[r]]);return a};let v=({title:e,content:a,prefixCls:i})=>e||a?t.createElement(t.Fragment,null,e&&t.createElement("div",{className:`${i}-title`},e),a&&t.createElement("div",{className:`${i}-inner-content`},a)):null,O=e=>{let{hashId:i,prefixCls:r,className:o,style:l,placement:s="top",title:p,content:g,children:u}=e,d=n(p),m=n(g),f=(0,a.default)(i,r,`${r}-pure`,`${r}-placement-${s}`,o);return t.createElement("div",{className:f,style:l},t.createElement("div",{className:`${r}-arrow`}),t.createElement(c.Popup,Object.assign({},e,{className:i,prefixCls:r}),u||t.createElement(v,{prefixCls:r,title:d,content:m})))},I=e=>{let{prefixCls:i,className:r}=e,n=A(e,["prefixCls","className"]),{getPrefixCls:o}=t.useContext(s.ConfigContext),l=o("popover",i),[p,c,g]=b(l);return p(t.createElement(O,Object.assign({},n,{prefixCls:l,hashId:c,className:(0,a.default)(r,g)})))};e.s(["Overlay",0,v,"default",0,I],310730);var $=function(e,t){var a={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(a[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,i=Object.getOwnPropertySymbols(e);r<i.length;r++)0>t.indexOf(i[r])&&Object.prototype.propertyIsEnumerable.call(e,i[r])&&(a[i[r]]=e[i[r]]);return a};let E=t.forwardRef((e,c)=>{var g,u;let{prefixCls:d,title:m,content:f,overlayClassName:_,placement:h="top",trigger:A="hover",children:O,mouseEnterDelay:I=.1,mouseLeaveDelay:E=.1,onOpenChange:C,overlayStyle:y={},styles:x,classNames:T}=e,k=$(e,["prefixCls","title","content","overlayClassName","placement","trigger","children","mouseEnterDelay","mouseLeaveDelay","onOpenChange","overlayStyle","styles","classNames"]),{getPrefixCls:w,className:L,style:S,classNames:M,styles:R}=(0,s.useComponentConfig)("popover"),j=w("popover",d),[N,P,D]=b(j),z=w(),H=(0,a.default)(_,P,D,L,M.root,null==T?void 0:T.root),B=(0,a.default)(M.body,null==T?void 0:T.body),[G,V]=(0,i.default)(!1,{value:null!=(g=e.open)?g:e.visible,defaultValue:null!=(u=e.defaultOpen)?u:e.defaultVisible}),F=(e,t)=>{V(e,!0),null==C||C(e,t)},q=n(m),U=n(f);return N(t.createElement(p.default,Object.assign({placement:h,trigger:A,mouseEnterDelay:I,mouseLeaveDelay:E},k,{prefixCls:j,classNames:{root:H,body:B},styles:{root:Object.assign(Object.assign(Object.assign(Object.assign({},R.root),S),y),null==x?void 0:x.root),body:Object.assign(Object.assign({},R.body),null==x?void 0:x.body)},ref:c,open:G,onOpenChange:e=>{F(e)},overlay:q||U?t.createElement(v,{prefixCls:j,title:q,content:U}):null,transitionName:(0,o.getTransitionName)(z,"zoom-big",k.transitionName),"data-popover-inject":!0}),(0,l.cloneElement)(O,{onKeyDown:e=>{var a,i;(0,t.isValidElement)(O)&&(null==(i=null==O?void 0:(a=O.props).onKeyDown)||i.call(a,e)),e.keyCode===r.default.ESC&&F(!1,e)}})))});E._InternalPanelDoNotUseOrYouWillBeFired=I,e.s(["default",0,E],829672),e.s(["Popover",0,E],282786)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var r=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(r.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["ArrowLeftOutlined",0,n],447566)},492030,e=>{"use strict";var t=e.i(121229);e.s(["CheckOutlined",()=>t.default])},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var r=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(r.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["LinkOutlined",0,n],596239)},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},62478,e=>{"use strict";var t=e.i(764205);let a=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,a])},818581,(e,t,a)=>{"use strict";Object.defineProperty(a,"__esModule",{value:!0}),Object.defineProperty(a,"useMergedRef",{enumerable:!0,get:function(){return r}});let i=e.r(271645);function r(e,t){let a=(0,i.useRef)(null),r=(0,i.useRef)(null);return(0,i.useCallback)(i=>{if(null===i){let e=a.current;e&&(a.current=null,e());let t=r.current;t&&(r.current=null,t())}else e&&(a.current=n(e,i)),t&&(r.current=n(t,i))},[e,t])}function n(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let a=e(t);return"function"==typeof a?a:()=>e(null)}}("function"==typeof a.default||"object"==typeof a.default&&null!==a.default)&&void 0===a.default.__esModule&&(Object.defineProperty(a.default,"__esModule",{value:!0}),Object.assign(a.default,a),t.exports=a.default)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var r=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(r.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["SafetyOutlined",0,n],602073)},44121,186515,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M408 442h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8zm-8 204c0 4.4 3.6 8 8 8h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56zm504-486H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zm0 632H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zM115.4 518.9L271.7 642c5.8 4.6 14.4.5 14.4-6.9V388.9c0-7.4-8.5-11.5-14.4-6.9L115.4 505.1a8.74 8.74 0 000 13.8z"}}]},name:"menu-fold",theme:"outlined"};var r=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(r.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["MenuFoldOutlined",0,n],44121);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M408 442h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8zm-8 204c0 4.4 3.6 8 8 8h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56zm504-486H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zm0 632H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zM142.4 642.1L298.7 519a8.84 8.84 0 000-13.9L142.4 381.9c-5.8-4.6-14.4-.5-14.4 6.9v246.3a8.9 8.9 0 0014.4 7z"}}]},name:"menu-unfold",theme:"outlined"};var l=a.forwardRef(function(e,i){return a.createElement(r.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["MenuUnfoldOutlined",0,l],186515)},928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},283713,e=>{"use strict";var t=e.i(271645),a=e.i(764205),i=e.i(612256);let r="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,i.useUIConfig)(),n=e?.is_control_plane??!1,o=e?.workers??[],[l,s]=(0,t.useState)(()=>localStorage.getItem(r));(0,t.useEffect)(()=>{if(!l||0===o.length)return;let e=o.find(e=>e.worker_id===l);e&&(0,a.switchToWorkerUrl)(e.url)},[l,o]);let p=o.find(e=>e.worker_id===l)??null,c=(0,t.useCallback)(e=>{let t=o.find(t=>t.worker_id===e);t&&(s(e),localStorage.setItem(r,e),(0,a.switchToWorkerUrl)(t.url))},[o]);return{isControlPlane:n,workers:o,selectedWorkerId:l,selectedWorker:p,selectWorker:c,disconnectFromWorker:(0,t.useCallback)(()=>{s(null),localStorage.removeItem(r),(0,a.switchToWorkerUrl)(null)},[])}}])},295320,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M704 446H320c-4.4 0-8 3.6-8 8v402c0 4.4 3.6 8 8 8h384c4.4 0 8-3.6 8-8V454c0-4.4-3.6-8-8-8zm-328 64h272v117H376V510zm272 290H376V683h272v117z"}},{tag:"path",attrs:{d:"M424 748a32 32 0 1064 0 32 32 0 10-64 0zm0-178a32 32 0 1064 0 32 32 0 10-64 0z"}},{tag:"path",attrs:{d:"M811.4 368.9C765.6 248 648.9 162 512.2 162S258.8 247.9 213 368.8C126.9 391.5 63.5 470.2 64 563.6 64.6 668 145.6 752.9 247.6 762c4.7.4 8.7-3.3 8.7-8v-60.4c0-4-3-7.4-7-7.9-27-3.4-52.5-15.2-72.1-34.5-24-23.5-37.2-55.1-37.2-88.6 0-28 9.1-54.4 26.2-76.4 16.7-21.4 40.2-36.9 66.1-43.7l37.9-10 13.9-36.7c8.6-22.8 20.6-44.2 35.7-63.5 14.9-19.2 32.6-36 52.4-50 41.1-28.9 89.5-44.2 140-44.2s98.9 15.3 140 44.3c19.9 14 37.5 30.8 52.4 50 15.1 19.3 27.1 40.7 35.7 63.5l13.8 36.6 37.8 10c54.2 14.4 92.1 63.7 92.1 120 0 33.6-13.2 65.1-37.2 88.6-19.5 19.2-44.9 31.1-71.9 34.5-4 .5-6.9 3.9-6.9 7.9V754c0 4.7 4.1 8.4 8.8 8 101.7-9.2 182.5-94 183.2-198.2.6-93.4-62.7-172.1-148.6-194.9z"}}]},name:"cloud-server",theme:"outlined"};var r=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(r.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["CloudServerOutlined",0,n],295320)},190272,785913,e=>{"use strict";var t,a,i=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),r=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a.REALTIME="realtime",a.INTERACTIONS="interactions",a);let n={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>r,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(i).includes(e)){let t=n[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:i,apiKey:n,inputMessage:o,chatHistory:l,selectedTags:s,selectedVectorStores:p,selectedGuardrails:c,selectedPolicies:g,selectedMCPServers:u,mcpServers:d,mcpServerToolRestrictions:m,selectedVoice:f,endpointType:_,selectedModel:h,selectedSdk:b,proxySettings:A}=e,v="session"===a?i:n,O=window.location.origin,I=A?.LITELLM_UI_API_DOC_BASE_URL;I&&I.trim()?O=I:A?.PROXY_BASE_URL&&(O=A.PROXY_BASE_URL);let $=o||"Your prompt here",E=$.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),C=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),y={};s.length>0&&(y.tags=s),p.length>0&&(y.vector_stores=p),c.length>0&&(y.guardrails=c),g.length>0&&(y.policies=g);let x=h||"your-model-name",T="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${O}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${O}"
)`;switch(_){case r.CHAT:{let e=Object.keys(y).length>0,a="";if(e){let e=JSON.stringify({metadata:y},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let i=C.length>0?C:[{role:"user",content:$}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${x}",
    messages=${JSON.stringify(i,null,4)}${a}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${x}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${E}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${a}
# )
# print(response_with_file)
`;break}case r.RESPONSES:{let e=Object.keys(y).length>0,a="";if(e){let e=JSON.stringify({metadata:y},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let i=C.length>0?C:[{role:"user",content:$}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${x}",
    input=${JSON.stringify(i,null,4)}${a}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${x}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${E}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${a}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===b?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${x}",
	prompt="${o}",
	n=1
)

json_response = json.loads(result.model_dump_json())

# Set the directory for the stored image
image_dir = os.path.join(os.curdir, 'images')

# If the directory doesn't exist, create it
if not os.path.isdir(image_dir):
	os.mkdir(image_dir)

# Initialize the image path
image_filename = f"generated_image_{int(time.time())}.png"
image_path = os.path.join(image_dir, image_filename)

try:
	# Retrieve the generated image
	if json_response.get("data") && len(json_response["data"]) > 0 && json_response["data"][0].get("url"):
			image_url = json_response["data"][0]["url"]
			generated_image = requests.get(image_url).content
			with open(image_path, "wb") as image_file:
					image_file.write(generated_image)

			print(f"Image saved to {image_path}")
			# Display the image
			image = Image.open(image_path)
			image.show()
	else:
			print("Could not find image URL in response.")
			print("Full response:", json_response)
except Exception as e:
	print(f"An error occurred: {e}")
	print("Full response:", json_response)
`:`
import base64
import os
import time
import json
from PIL import Image
import requests

# Helper function to encode images to base64
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to create a file (simplified for this example)
def create_file(image_path):
	# In a real implementation, this would upload the file to OpenAI
	# For this example, we'll just return a placeholder ID
	return f"file_{os.path.basename(image_path).replace('.', '_')}"

# The prompt entered by the user
prompt = "${E}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${x}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
							},
							{
									"type": "input_image",
									"file_id": file_id2,
							}
					],
			}
	],
	tools=[{"type": "image_generation"}],
)

# Process the response
image_generation_calls = [
	output
	for output in response.output
	if output.type == "image_generation_call"
]

image_data = [output.result for output in image_generation_calls]

if image_data:
	image_base64 = image_data[0]
	image_filename = f"edited_image_{int(time.time())}.png"
	with open(image_filename, "wb") as f:
			f.write(base64.b64decode(image_base64))
	print(f"Image saved to {image_filename}")
else:
	# If no image is generated, there might be a text response with an explanation
	text_response = [output.text for output in response.output if hasattr(output, 'text')]
	if text_response:
			print("No image generated. Model response:")
			print("\\n".join(text_response))
	else:
			print("No image data found in response.")
	print("Full response for debugging:")
	print(response)
`;break;case r.IMAGE_EDITS:t="azure"===b?`
import base64
import os
import time
import json
from PIL import Image
import requests

# Helper function to encode images to base64
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

# The prompt entered by the user
prompt = "${E}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${x}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
							},
							{
									"type": "input_image",
									"file_id": file_id2,
							}
					],
			}
	],
	tools=[{"type": "image_generation"}],
)

# Process the response
image_generation_calls = [
	output
	for output in response.output
	if output.type == "image_generation_call"
]

image_data = [output.result for output in image_generation_calls]

if image_data:
	image_base64 = image_data[0]
	image_filename = f"edited_image_{int(time.time())}.png"
	with open(image_filename, "wb") as f:
			f.write(base64.b64decode(image_base64))
	print(f"Image saved to {image_filename}")
else:
	# If no image is generated, there might be a text response with an explanation
	text_response = [output.text for output in response.output if hasattr(output, 'text')]
	if text_response:
			print("No image generated. Model response:")
			print("\\n".join(text_response))
	else:
			print("No image data found in response.")
	print("Full response for debugging:")
	print(response)
`:`
import base64
import os
import time

# Helper function to encode images to base64
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to create a file (simplified for this example)
def create_file(image_path):
	# In a real implementation, this would upload the file to OpenAI
	# For this example, we'll just return a placeholder ID
	return f"file_{os.path.basename(image_path).replace('.', '_')}"

# The prompt entered by the user
prompt = "${E}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${x}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
							},
							{
									"type": "input_image",
									"file_id": file_id2,
							}
					],
			}
	],
	tools=[{"type": "image_generation"}],
)

# Process the response
image_generation_calls = [
	output
	for output in response.output
	if output.type == "image_generation_call"
]

image_data = [output.result for output in image_generation_calls]

if image_data:
	image_base64 = image_data[0]
	image_filename = f"edited_image_{int(time.time())}.png"
	with open(image_filename, "wb") as f:
			f.write(base64.b64decode(image_base64))
	print(f"Image saved to {image_filename}")
else:
	# If no image is generated, there might be a text response with an explanation
	text_response = [output.text for output in response.output if hasattr(output, 'text')]
	if text_response:
			print("No image generated. Model response:")
			print("\\n".join(text_response))
	else:
			print("No image data found in response.")
	print("Full response for debugging:")
	print(response)
`;break;case r.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${o||"Your string here"}",
	model="${x}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case r.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${x}",
	file=audio_file${o?`,
	prompt="${o.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case r.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${x}",
	input="${o||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${x}",
#     input="${o||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${T}
${t}`}],190272)}]);