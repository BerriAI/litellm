(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},916925,e=>{"use strict";var t,r=((t={}).A2A_Agent="A2A Agent",t.AIML="AI/ML API",t.Bedrock="Amazon Bedrock",t.Anthropic="Anthropic",t.AssemblyAI="AssemblyAI",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.Cerebras="Cerebras",t.Cohere="Cohere",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.ElevenLabs="ElevenLabs",t.FalAI="Fal AI",t.FireworksAI="Fireworks AI",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.Hosted_Vllm="vllm",t.Infinity="Infinity",t.JinaAI="Jina AI",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.Ollama="Ollama",t.OpenAI="OpenAI",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.Perplexity="Perplexity",t.RunwayML="RunwayML",t.Sambanova="Sambanova",t.Snowflake="Snowflake",t.TogetherAI="TogetherAI",t.Triton="Triton",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.xAI="xAI",t.SAP="SAP Generative AI Hub",t.Watsonx="Watsonx",t);let o={A2A_Agent:"a2a_agent",AIML:"aiml",OpenAI:"openai",OpenAI_Text:"text-completion-openai",Azure:"azure",Azure_AI_Studio:"azure_ai",Anthropic:"anthropic",Google_AI_Studio:"gemini",Bedrock:"bedrock",Groq:"groq",MiniMax:"minimax",MistralAI:"mistral",Cohere:"cohere",OpenAI_Compatible:"openai",OpenAI_Text_Compatible:"text-completion-openai",Vertex_AI:"vertex_ai",Databricks:"databricks",Dashscope:"dashscope",xAI:"xai",Deepseek:"deepseek",Ollama:"ollama",AssemblyAI:"assemblyai",Cerebras:"cerebras",Sambanova:"sambanova",Perplexity:"perplexity",RunwayML:"runwayml",TogetherAI:"together_ai",Openrouter:"openrouter",Oracle:"oci",Snowflake:"snowflake",FireworksAI:"fireworks_ai",GradientAI:"gradient_ai",Triton:"triton",Deepgram:"deepgram",ElevenLabs:"elevenlabs",FalAI:"fal_ai",SageMaker:"sagemaker_chat",Voyage:"voyage",JinaAI:"jina_ai",VolcEngine:"volcengine",DeepInfra:"deepinfra",Hosted_Vllm:"hosted_vllm",Infinity:"infinity",SAP:"sap",Watsonx:"watsonx"},a="../ui/assets/logos/",i={"A2A Agent":`${a}a2a_agent.png`,"AI/ML API":`${a}aiml_api.svg`,Anthropic:`${a}anthropic.svg`,AssemblyAI:`${a}assemblyai_small.png`,Azure:`${a}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${a}microsoft_azure.svg`,"Amazon Bedrock":`${a}bedrock.svg`,"AWS SageMaker":`${a}bedrock.svg`,Cerebras:`${a}cerebras.svg`,Cohere:`${a}cohere.svg`,"Databricks (Qwen API)":`${a}databricks.svg`,Dashscope:`${a}dashscope.svg`,Deepseek:`${a}deepseek.svg`,"Fireworks AI":`${a}fireworks.svg`,Groq:`${a}groq.svg`,"Google AI Studio":`${a}google.svg`,vllm:`${a}vllm.png`,Infinity:`${a}infinity.png`,MiniMax:`${a}minimax.svg`,"Mistral AI":`${a}mistral.svg`,Ollama:`${a}ollama.svg`,OpenAI:`${a}openai_small.svg`,"OpenAI Text Completion":`${a}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${a}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${a}openai_small.svg`,Openrouter:`${a}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${a}oracle.svg`,Perplexity:`${a}perplexity-ai.svg`,RunwayML:`${a}runwayml.png`,Sambanova:`${a}sambanova.svg`,Snowflake:`${a}snowflake.svg`,TogetherAI:`${a}togetherai.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${a}google.svg`,xAI:`${a}xai.svg`,GradientAI:`${a}gradientai.svg`,Triton:`${a}nvidia_triton.png`,Deepgram:`${a}deepgram.png`,ElevenLabs:`${a}elevenlabs.png`,"Fal AI":`${a}fal_ai.jpg`,"Voyage AI":`${a}voyage.webp`,"Jina AI":`${a}jina.png`,VolcEngine:`${a}volcengine.png`,DeepInfra:`${a}deepinfra.png`,"SAP Generative AI Hub":`${a}sap.png`};e.s(["Providers",()=>r,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:i[e],displayName:e}}let t=Object.keys(o).find(t=>o[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let a=r[t];return{logo:i[a],displayName:a}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let r=o[e];console.log(`Provider mapped to: ${r}`);let a=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let o=t.litellm_provider;(o===r||"string"==typeof o&&o.includes(r))&&a.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&a.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&a.push(e)}))),a},"providerLogoMap",0,i,"provider_map",0,o])},94629,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,r],94629)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},292639,e=>{"use strict";var t=e.i(764205),r=e.i(266027);let o=(0,e.i(243652).createQueryKeys)("uiSettings");e.s(["useUISettings",0,()=>(0,r.useQuery)({queryKey:o.list({}),queryFn:async()=>await (0,t.getUiSettings)(),staleTime:36e5,gcTime:36e5})])},502547,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["ChevronRightIcon",0,r],502547)},262218,e=>{"use strict";e.i(247167);var t=e.i(271645),r=e.i(343794),o=e.i(529681),a=e.i(702779),i=e.i(563113),n=e.i(763731),s=e.i(121872),l=e.i(242064);e.i(296059);var c=e.i(915654);e.i(262370);var d=e.i(135551),u=e.i(183293),g=e.i(246422),p=e.i(838378);let m=e=>{let{lineWidth:t,fontSizeIcon:r,calc:o}=e,a=e.fontSizeSM;return(0,p.mergeToken)(e,{tagFontSize:a,tagLineHeight:(0,c.unit)(o(e.lineHeightSM).mul(a).equal()),tagIconSize:o(r).sub(o(t).mul(2)).equal(),tagPaddingHorizontal:8,tagBorderlessBg:e.defaultBg})},f=e=>({defaultBg:new d.FastColor(e.colorFillQuaternary).onBackground(e.colorBgContainer).toHexString(),defaultColor:e.colorText}),h=(0,g.genStyleHooks)("Tag",e=>(e=>{let{paddingXXS:t,lineWidth:r,tagPaddingHorizontal:o,componentCls:a,calc:i}=e,n=i(o).sub(r).equal(),s=i(t).sub(r).equal();return{[a]:Object.assign(Object.assign({},(0,u.resetComponent)(e)),{display:"inline-block",height:"auto",marginInlineEnd:e.marginXS,paddingInline:n,fontSize:e.tagFontSize,lineHeight:e.tagLineHeight,whiteSpace:"nowrap",background:e.defaultBg,border:`${(0,c.unit)(e.lineWidth)} ${e.lineType} ${e.colorBorder}`,borderRadius:e.borderRadiusSM,opacity:1,transition:`all ${e.motionDurationMid}`,textAlign:"start",position:"relative",[`&${a}-rtl`]:{direction:"rtl"},"&, a, a:hover":{color:e.defaultColor},[`${a}-close-icon`]:{marginInlineStart:s,fontSize:e.tagIconSize,color:e.colorIcon,cursor:"pointer",transition:`all ${e.motionDurationMid}`,"&:hover":{color:e.colorTextHeading}},[`&${a}-has-color`]:{borderColor:"transparent",[`&, a, a:hover, ${e.iconCls}-close, ${e.iconCls}-close:hover`]:{color:e.colorTextLightSolid}},"&-checkable":{backgroundColor:"transparent",borderColor:"transparent",cursor:"pointer",[`&:not(${a}-checkable-checked):hover`]:{color:e.colorPrimary,backgroundColor:e.colorFillSecondary},"&:active, &-checked":{color:e.colorTextLightSolid},"&-checked":{backgroundColor:e.colorPrimary,"&:hover":{backgroundColor:e.colorPrimaryHover}},"&:active":{backgroundColor:e.colorPrimaryActive}},"&-hidden":{display:"none"},[`> ${e.iconCls} + span, > span + ${e.iconCls}`]:{marginInlineStart:n}}),[`${a}-borderless`]:{borderColor:"transparent",background:e.tagBorderlessBg}}})(m(e)),f);var b=function(e,t){var r={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(r[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var a=0,o=Object.getOwnPropertySymbols(e);a<o.length;a++)0>t.indexOf(o[a])&&Object.prototype.propertyIsEnumerable.call(e,o[a])&&(r[o[a]]=e[o[a]]);return r};let _=t.forwardRef((e,o)=>{let{prefixCls:a,style:i,className:n,checked:s,children:c,icon:d,onChange:u,onClick:g}=e,p=b(e,["prefixCls","style","className","checked","children","icon","onChange","onClick"]),{getPrefixCls:m,tag:f}=t.useContext(l.ConfigContext),_=m("tag",a),[v,y,x]=h(_),w=(0,r.default)(_,`${_}-checkable`,{[`${_}-checkable-checked`]:s},null==f?void 0:f.className,n,y,x);return v(t.createElement("span",Object.assign({},p,{ref:o,style:Object.assign(Object.assign({},i),null==f?void 0:f.style),className:w,onClick:e=>{null==u||u(!s),null==g||g(e)}}),d,t.createElement("span",null,c)))});var v=e.i(403541);let y=(0,g.genSubStyleComponent)(["Tag","preset"],e=>{let t;return t=m(e),(0,v.genPresetColor)(t,(e,{textColor:r,lightBorderColor:o,lightColor:a,darkColor:i})=>({[`${t.componentCls}${t.componentCls}-${e}`]:{color:r,background:a,borderColor:o,"&-inverse":{color:t.colorTextLightSolid,background:i,borderColor:i},[`&${t.componentCls}-borderless`]:{borderColor:"transparent"}}}))},f),x=(e,t,r)=>{let o="string"!=typeof r?r:r.charAt(0).toUpperCase()+r.slice(1);return{[`${e.componentCls}${e.componentCls}-${t}`]:{color:e[`color${r}`],background:e[`color${o}Bg`],borderColor:e[`color${o}Border`],[`&${e.componentCls}-borderless`]:{borderColor:"transparent"}}}},w=(0,g.genSubStyleComponent)(["Tag","status"],e=>{let t=m(e);return[x(t,"success","Success"),x(t,"processing","Info"),x(t,"error","Error"),x(t,"warning","Warning")]},f);var C=function(e,t){var r={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(r[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var a=0,o=Object.getOwnPropertySymbols(e);a<o.length;a++)0>t.indexOf(o[a])&&Object.prototype.propertyIsEnumerable.call(e,o[a])&&(r[o[a]]=e[o[a]]);return r};let I=t.forwardRef((e,c)=>{let{prefixCls:d,className:u,rootClassName:g,style:p,children:m,icon:f,color:b,onClose:_,bordered:v=!0,visible:x}=e,I=C(e,["prefixCls","className","rootClassName","style","children","icon","color","onClose","bordered","visible"]),{getPrefixCls:k,direction:A,tag:S}=t.useContext(l.ConfigContext),[$,E]=t.useState(!0),O=(0,o.default)(I,["closeIcon","closable"]);t.useEffect(()=>{void 0!==x&&E(x)},[x]);let j=(0,a.isPresetColor)(b),T=(0,a.isPresetStatusColor)(b),M=j||T,P=Object.assign(Object.assign({backgroundColor:b&&!M?b:void 0},null==S?void 0:S.style),p),N=k("tag",d),[L,R,z]=h(N),D=(0,r.default)(N,null==S?void 0:S.className,{[`${N}-${b}`]:M,[`${N}-has-color`]:b&&!M,[`${N}-hidden`]:!$,[`${N}-rtl`]:"rtl"===A,[`${N}-borderless`]:!v},u,g,R,z),H=e=>{e.stopPropagation(),null==_||_(e),e.defaultPrevented||E(!1)},[,B]=(0,i.useClosable)((0,i.pickClosable)(e),(0,i.pickClosable)(S),{closable:!1,closeIconRender:e=>{let o=t.createElement("span",{className:`${N}-close-icon`,onClick:H},e);return(0,n.replaceElement)(e,o,e=>({onClick:t=>{var r;null==(r=null==e?void 0:e.onClick)||r.call(e,t),H(t)},className:(0,r.default)(null==e?void 0:e.className,`${N}-close-icon`)}))}}),G="function"==typeof I.onClick||m&&"a"===m.type,V=f||null,F=V?t.createElement(t.Fragment,null,V,m&&t.createElement("span",null,m)):m,U=t.createElement("span",Object.assign({},O,{ref:c,className:D,style:P}),F,B,j&&t.createElement(y,{key:"preset",prefixCls:N}),T&&t.createElement(w,{key:"status",prefixCls:N}));return L(G?t.createElement(s.default,{component:"Tag"},U):U)});I.CheckableTag=_,e.s(["Tag",0,I],262218)},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},991124,e=>{"use strict";let t=(0,e.i(475254).default)("copy",[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]]);e.s(["default",()=>t])},250980,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlusCircleIcon",0,r],250980)},434626,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,r],434626)},902555,e=>{"use strict";var t=e.i(843476),r=e.i(591935),o=e.i(122577),a=e.i(278587),i=e.i(68155),n=e.i(360820),s=e.i(871943),l=e.i(434626),c=e.i(592968),d=e.i(115504),u=e.i(752978);function g({icon:e,onClick:r,className:o,disabled:a,dataTestId:i}){return a?(0,t.jsx)(u.Icon,{icon:e,size:"sm",className:"opacity-50 cursor-not-allowed","data-testid":i}):(0,t.jsx)(u.Icon,{icon:e,size:"sm",onClick:r,className:(0,d.cx)("cursor-pointer",o),"data-testid":i})}let p={Edit:{icon:r.PencilAltIcon,className:"hover:text-blue-600"},Delete:{icon:i.TrashIcon,className:"hover:text-red-600"},Test:{icon:o.PlayIcon,className:"hover:text-blue-600"},Regenerate:{icon:a.RefreshIcon,className:"hover:text-green-600"},Up:{icon:n.ChevronUpIcon,className:"hover:text-blue-600"},Down:{icon:s.ChevronDownIcon,className:"hover:text-blue-600"},Open:{icon:l.ExternalLinkIcon,className:"hover:text-green-600"}};function m({onClick:e,tooltipText:r,disabled:o=!1,disabledTooltipText:a,dataTestId:i,variant:n}){let{icon:s,className:l}=p[n];return(0,t.jsx)(c.Tooltip,{title:o?a:r,children:(0,t.jsx)("span",{children:(0,t.jsx)(g,{icon:s,onClick:e,className:l,disabled:o,dataTestId:i})})})}e.s(["default",()=>m],902555)},122577,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"}),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlayIcon",0,r],122577)},278587,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"}))});e.s(["RefreshIcon",0,r],278587)},207670,e=>{"use strict";function t(){for(var e,t,r=0,o="",a=arguments.length;r<a;r++)(e=arguments[r])&&(t=function e(t){var r,o,a="";if("string"==typeof t||"number"==typeof t)a+=t;else if("object"==typeof t)if(Array.isArray(t)){var i=t.length;for(r=0;r<i;r++)t[r]&&(o=e(t[r]))&&(a&&(a+=" "),a+=o)}else for(o in t)t[o]&&(a&&(a+=" "),a+=o);return a}(e))&&(o&&(o+=" "),o+=t);return o}e.s(["clsx",()=>t,"default",0,t])},728889,e=>{"use strict";var t=e.i(290571),r=e.i(271645),o=e.i(829087),a=e.i(480731),i=e.i(444755),n=e.i(673706),s=e.i(95779);let l={xs:{paddingX:"px-1.5",paddingY:"py-1.5"},sm:{paddingX:"px-1.5",paddingY:"py-1.5"},md:{paddingX:"px-2",paddingY:"py-2"},lg:{paddingX:"px-2",paddingY:"py-2"},xl:{paddingX:"px-2.5",paddingY:"py-2.5"}},c={xs:{height:"h-3",width:"w-3"},sm:{height:"h-5",width:"w-5"},md:{height:"h-5",width:"w-5"},lg:{height:"h-7",width:"w-7"},xl:{height:"h-9",width:"w-9"}},d={simple:{rounded:"",border:"",ring:"",shadow:""},light:{rounded:"rounded-tremor-default",border:"",ring:"",shadow:""},shadow:{rounded:"rounded-tremor-default",border:"border",ring:"",shadow:"shadow-tremor-card dark:shadow-dark-tremor-card"},solid:{rounded:"rounded-tremor-default",border:"border-2",ring:"ring-1",shadow:""},outlined:{rounded:"rounded-tremor-default",border:"border",ring:"ring-2",shadow:""}},u=(0,n.makeClassName)("Icon"),g=r.default.forwardRef((e,g)=>{let{icon:p,variant:m="simple",tooltip:f,size:h=a.Sizes.SM,color:b,className:_}=e,v=(0,t.__rest)(e,["icon","variant","tooltip","size","color","className"]),y=((e,t)=>{switch(e){case"simple":return{textColor:t?(0,n.getColorClassNames)(t,s.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:"",borderColor:"",ringColor:""};case"light":return{textColor:t?(0,n.getColorClassNames)(t,s.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,s.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-brand-muted dark:bg-dark-tremor-brand-muted",borderColor:"",ringColor:""};case"shadow":return{textColor:t?(0,n.getColorClassNames)(t,s.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,s.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-background dark:bg-dark-tremor-background",borderColor:"border-tremor-border dark:border-dark-tremor-border",ringColor:""};case"solid":return{textColor:t?(0,n.getColorClassNames)(t,s.colorPalette.text).textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",bgColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,s.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-brand dark:bg-dark-tremor-brand",borderColor:"border-tremor-brand-inverted dark:border-dark-tremor-brand-inverted",ringColor:"ring-tremor-ring dark:ring-dark-tremor-ring"};case"outlined":return{textColor:t?(0,n.getColorClassNames)(t,s.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,s.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-background dark:bg-dark-tremor-background",borderColor:t?(0,n.getColorClassNames)(t,s.colorPalette.ring).borderColor:"border-tremor-brand-subtle dark:border-dark-tremor-brand-subtle",ringColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,s.colorPalette.ring).ringColor,"ring-opacity-40"):"ring-tremor-brand-muted dark:ring-dark-tremor-brand-muted"}}})(m,b),{tooltipProps:x,getReferenceProps:w}=(0,o.useTooltip)();return r.default.createElement("span",Object.assign({ref:(0,n.mergeRefs)([g,x.refs.setReference]),className:(0,i.tremorTwMerge)(u("root"),"inline-flex shrink-0 items-center justify-center",y.bgColor,y.textColor,y.borderColor,y.ringColor,d[m].rounded,d[m].border,d[m].shadow,d[m].ring,l[h].paddingX,l[h].paddingY,_)},w,v),r.default.createElement(o.default,Object.assign({text:f},x)),r.default.createElement(p,{className:(0,i.tremorTwMerge)(u("icon"),"shrink-0",c[h].height,c[h].width)}))});g.displayName="Icon",e.s(["default",()=>g],728889)},752978,e=>{"use strict";var t=e.i(728889);e.s(["Icon",()=>t.default])},591935,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"}))});e.s(["PencilAltIcon",0,r],591935)},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["CrownOutlined",0,i],100486)},209261,e=>{"use strict";e.s(["extractCategories",0,e=>{let t=new Set;return e.forEach(e=>{e.category&&""!==e.category.trim()&&t.add(e.category)}),["All",...Array.from(t).sort(),"Other"]},"filterPluginsByCategory",0,(e,t)=>"All"===t?e:"Other"===t?e.filter(e=>!e.category||""===e.category.trim()):e.filter(e=>e.category===t),"filterPluginsBySearch",0,(e,t)=>{if(!t||""===t.trim())return e;let r=t.toLowerCase().trim();return e.filter(e=>{let t=e.name.toLowerCase().includes(r),o=e.description?.toLowerCase().includes(r)||!1,a=e.keywords?.some(e=>e.toLowerCase().includes(r))||!1;return t||o||a})},"formatDateString",0,e=>{if(!e)return"N/A";try{return new Date(e).toLocaleDateString("en-US",{year:"numeric",month:"short",day:"numeric"})}catch(e){return"Invalid date"}},"formatInstallCommand",0,e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"url"===e.source&&e.url?e.url:"Unknown source","getSourceLink",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:"url"===e.source&&e.url?e.url:null,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)])},771674,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["UserOutlined",0,i],771674)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["SafetyOutlined",0,i],602073)},818581,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"useMergedRef",{enumerable:!0,get:function(){return a}});let o=e.r(271645);function a(e,t){let r=(0,o.useRef)(null),a=(0,o.useRef)(null);return(0,o.useCallback)(o=>{if(null===o){let e=r.current;e&&(r.current=null,e());let t=a.current;t&&(a.current=null,t())}else e&&(r.current=i(e,o)),t&&(a.current=i(t,o))},[e,t])}function i(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let r=e(t);return"function"==typeof r?r:()=>e(null)}}("function"==typeof r.default||"object"==typeof r.default&&null!==r.default)&&void 0===r.default.__esModule&&(Object.defineProperty(r.default,"__esModule",{value:!0}),Object.assign(r.default,r),t.exports=r.default)},62478,e=>{"use strict";var t=e.i(764205);let r=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,r])},190272,785913,e=>{"use strict";var t,r,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),a=((r={}).IMAGE="image",r.VIDEO="video",r.CHAT="chat",r.RESPONSES="responses",r.IMAGE_EDITS="image_edits",r.ANTHROPIC_MESSAGES="anthropic_messages",r.EMBEDDINGS="embeddings",r.SPEECH="speech",r.TRANSCRIPTION="transcription",r.A2A_AGENTS="a2a_agents",r.MCP="mcp",r);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>a,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(o).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:r,accessToken:o,apiKey:i,inputMessage:n,chatHistory:s,selectedTags:l,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:u,selectedMCPServers:g,mcpServers:p,mcpServerToolRestrictions:m,selectedVoice:f,endpointType:h,selectedModel:b,selectedSdk:_,proxySettings:v}=e,y="session"===r?o:i,x=window.location.origin,w=v?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?x=w:v?.PROXY_BASE_URL&&(x=v.PROXY_BASE_URL);let C=n||"Your prompt here",I=C.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),k=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),A={};l.length>0&&(A.tags=l),c.length>0&&(A.vector_stores=c),d.length>0&&(A.guardrails=d),u.length>0&&(A.policies=u);let S=b||"your-model-name",$="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${x}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${x}"
)`;switch(h){case a.CHAT:{let e=Object.keys(A).length>0,r="";if(e){let e=JSON.stringify({metadata:A},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let o=k.length>0?k:[{role:"user",content:C}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${S}",
    messages=${JSON.stringify(o,null,4)}${r}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${S}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${I}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${r}
# )
# print(response_with_file)
`;break}case a.RESPONSES:{let e=Object.keys(A).length>0,r="";if(e){let e=JSON.stringify({metadata:A},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let o=k.length>0?k:[{role:"user",content:C}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${S}",
    input=${JSON.stringify(o,null,4)}${r}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${S}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${I}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${r}
# )
# print(response_with_file.output_text)
`;break}case a.IMAGE:t="azure"===_?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${S}",
	prompt="${n}",
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
prompt = "${I}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${S}",
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
`;break;case a.IMAGE_EDITS:t="azure"===_?`
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
prompt = "${I}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${S}",
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
prompt = "${I}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${S}",
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
`;break;case a.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${S}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case a.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${S}",
	file=audio_file${n?`,
	prompt="${n.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case a.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${S}",
	input="${n||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${S}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${$}
${t}`}],190272)},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},275144,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(764205);let a=(0,r.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:i})=>{let[n,s]=(0,r.useState)(null);return(0,r.useEffect)(()=>{(async()=>{try{let e=(0,o.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",r=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(r.ok){let e=await r.json();e.values?.logo_url&&s(e.values.logo_url)}}catch(e){console.warn("Failed to load logo settings from backend:",e)}})()},[]),(0,t.jsx)(a.Provider,{value:{logoUrl:n,setLogoUrl:s},children:e})},"useTheme",0,()=>{let e=(0,r.useContext)(a);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},115571,371401,e=>{"use strict";let t="local-storage-change";function r(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function o(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function a(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function i(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>r,"getLocalStorageItem",()=>o,"removeLocalStorageItem",()=>i,"setLocalStorageItem",()=>a],115571);var n=e.i(271645);function s(e){let r=t=>{"disableUsageIndicator"===t.key&&e()},o=t=>{let{key:r}=t.detail;"disableUsageIndicator"===r&&e()};return window.addEventListener("storage",r),window.addEventListener(t,o),()=>{window.removeEventListener("storage",r),window.removeEventListener(t,o)}}function l(){return"true"===o("disableUsageIndicator")}function c(){return(0,n.useSyncExternalStore)(s,l)}e.s(["useDisableUsageIndicator",()=>c],371401)},798496,e=>{"use strict";var t=e.i(843476),r=e.i(152990),o=e.i(682830),a=e.i(271645),i=e.i(269200),n=e.i(427612),s=e.i(64848),l=e.i(942232),c=e.i(496020),d=e.i(977572),u=e.i(94629),g=e.i(360820),p=e.i(871943);function m({data:e=[],columns:m,isLoading:f=!1,defaultSorting:h=[],pagination:b,onPaginationChange:_,enablePagination:v=!1}){let[y,x]=a.default.useState(h),[w]=a.default.useState("onChange"),[C,I]=a.default.useState({}),[k,A]=a.default.useState({}),S=(0,r.useReactTable)({data:e,columns:m,state:{sorting:y,columnSizing:C,columnVisibility:k,...v&&b?{pagination:b}:{}},columnResizeMode:w,onSortingChange:x,onColumnSizingChange:I,onColumnVisibilityChange:A,...v&&_?{onPaginationChange:_}:{},getCoreRowModel:(0,o.getCoreRowModel)(),getSortedRowModel:(0,o.getSortedRowModel)(),...v?{getPaginationRowModel:(0,o.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(i.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:S.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(n.TableHead,{children:S.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(s.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,r.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(g.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(p.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(u.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(l.TableBody,{children:f?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:m.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):S.getRowModel().rows.length>0?S.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,r.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:m.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>m])}]);