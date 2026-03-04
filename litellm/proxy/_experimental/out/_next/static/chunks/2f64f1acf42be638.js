(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,916925,e=>{"use strict";var t,a=((t={}).A2A_Agent="A2A Agent",t.AIML="AI/ML API",t.Bedrock="Amazon Bedrock",t.Anthropic="Anthropic",t.AssemblyAI="AssemblyAI",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.Cerebras="Cerebras",t.Cohere="Cohere",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.ElevenLabs="ElevenLabs",t.FalAI="Fal AI",t.FireworksAI="Fireworks AI",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.Hosted_Vllm="vllm",t.Infinity="Infinity",t.JinaAI="Jina AI",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.Ollama="Ollama",t.OpenAI="OpenAI",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.Perplexity="Perplexity",t.RunwayML="RunwayML",t.Sambanova="Sambanova",t.Snowflake="Snowflake",t.TogetherAI="TogetherAI",t.Triton="Triton",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.xAI="xAI",t.SAP="SAP Generative AI Hub",t.Watsonx="Watsonx",t);let i={A2A_Agent:"a2a_agent",AIML:"aiml",OpenAI:"openai",OpenAI_Text:"text-completion-openai",Azure:"azure",Azure_AI_Studio:"azure_ai",Anthropic:"anthropic",Google_AI_Studio:"gemini",Bedrock:"bedrock",Groq:"groq",MiniMax:"minimax",MistralAI:"mistral",Cohere:"cohere",OpenAI_Compatible:"openai",OpenAI_Text_Compatible:"text-completion-openai",Vertex_AI:"vertex_ai",Databricks:"databricks",Dashscope:"dashscope",xAI:"xai",Deepseek:"deepseek",Ollama:"ollama",AssemblyAI:"assemblyai",Cerebras:"cerebras",Sambanova:"sambanova",Perplexity:"perplexity",RunwayML:"runwayml",TogetherAI:"together_ai",Openrouter:"openrouter",Oracle:"oci",Snowflake:"snowflake",FireworksAI:"fireworks_ai",GradientAI:"gradient_ai",Triton:"triton",Deepgram:"deepgram",ElevenLabs:"elevenlabs",FalAI:"fal_ai",SageMaker:"sagemaker_chat",Voyage:"voyage",JinaAI:"jina_ai",VolcEngine:"volcengine",DeepInfra:"deepinfra",Hosted_Vllm:"hosted_vllm",Infinity:"infinity",SAP:"sap",Watsonx:"watsonx"},n="../ui/assets/logos/",r={"A2A Agent":`${n}a2a_agent.png`,"AI/ML API":`${n}aiml_api.svg`,Anthropic:`${n}anthropic.svg`,AssemblyAI:`${n}assemblyai_small.png`,Azure:`${n}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${n}microsoft_azure.svg`,"Amazon Bedrock":`${n}bedrock.svg`,"AWS SageMaker":`${n}bedrock.svg`,Cerebras:`${n}cerebras.svg`,Cohere:`${n}cohere.svg`,"Databricks (Qwen API)":`${n}databricks.svg`,Dashscope:`${n}dashscope.svg`,Deepseek:`${n}deepseek.svg`,"Fireworks AI":`${n}fireworks.svg`,Groq:`${n}groq.svg`,"Google AI Studio":`${n}google.svg`,vllm:`${n}vllm.png`,Infinity:`${n}infinity.png`,MiniMax:`${n}minimax.svg`,"Mistral AI":`${n}mistral.svg`,Ollama:`${n}ollama.svg`,OpenAI:`${n}openai_small.svg`,"OpenAI Text Completion":`${n}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${n}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${n}openai_small.svg`,Openrouter:`${n}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${n}oracle.svg`,Perplexity:`${n}perplexity-ai.svg`,RunwayML:`${n}runwayml.png`,Sambanova:`${n}sambanova.svg`,Snowflake:`${n}snowflake.svg`,TogetherAI:`${n}togetherai.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${n}google.svg`,xAI:`${n}xai.svg`,GradientAI:`${n}gradientai.svg`,Triton:`${n}nvidia_triton.png`,Deepgram:`${n}deepgram.png`,ElevenLabs:`${n}elevenlabs.png`,"Fal AI":`${n}fal_ai.jpg`,"Voyage AI":`${n}voyage.webp`,"Jina AI":`${n}jina.png`,VolcEngine:`${n}volcengine.png`,DeepInfra:`${n}deepinfra.png`,"SAP Generative AI Hub":`${n}sap.png`};e.s(["Providers",()=>a,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:r[e],displayName:e}}let t=Object.keys(i).find(t=>i[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let n=a[t];return{logo:r[n],displayName:n}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let a=i[e];console.log(`Provider mapped to: ${a}`);let n=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let i=t.litellm_provider;(i===a||"string"==typeof i&&i.includes(a))&&n.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&n.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&n.push(e)}))),n},"providerLogoMap",0,r,"provider_map",0,i])},629569,e=>{"use strict";var t=e.i(290571),a=e.i(95779),i=e.i(444755),n=e.i(673706),r=e.i(271645);let o=r.default.forwardRef((e,o)=>{let{color:l,children:s,className:c}=e,d=(0,t.__rest)(e,["color","children","className"]);return r.default.createElement("p",Object.assign({ref:o,className:(0,i.tremorTwMerge)("font-medium text-tremor-title",l?(0,n.getColorClassNames)(l,a.colorPalette.darkText).textColor:"text-tremor-content-strong dark:text-dark-tremor-content-strong",c)},d),s)});o.displayName="Title",e.s(["Title",()=>o],629569)},790848,e=>{"use strict";e.i(247167);var t=e.i(271645),a=e.i(739295),i=e.i(343794),n=e.i(931067),r=e.i(211577),o=e.i(392221),l=e.i(703923),s=e.i(914949),c=e.i(404948),d=["prefixCls","className","checked","defaultChecked","disabled","loadingIcon","checkedChildren","unCheckedChildren","onClick","onChange","onKeyDown"],p=t.forwardRef(function(e,a){var p,u=e.prefixCls,m=void 0===u?"rc-switch":u,g=e.className,f=e.checked,h=e.defaultChecked,_=e.disabled,b=e.loadingIcon,I=e.checkedChildren,v=e.unCheckedChildren,y=e.onClick,w=e.onChange,x=e.onKeyDown,S=(0,l.default)(e,d),$=(0,s.default)(!1,{value:f,defaultValue:h}),A=(0,o.default)($,2),k=A[0],E=A[1];function C(e,t){var a=k;return _||(E(a=e),null==w||w(a,t)),a}var O=(0,i.default)(m,g,(p={},(0,r.default)(p,"".concat(m,"-checked"),k),(0,r.default)(p,"".concat(m,"-disabled"),_),p));return t.createElement("button",(0,n.default)({},S,{type:"button",role:"switch","aria-checked":k,disabled:_,className:O,ref:a,onKeyDown:function(e){e.which===c.default.LEFT?C(!1,e):e.which===c.default.RIGHT&&C(!0,e),null==x||x(e)},onClick:function(e){var t=C(!k,e);null==y||y(t,e)}}),b,t.createElement("span",{className:"".concat(m,"-inner")},t.createElement("span",{className:"".concat(m,"-inner-checked")},I),t.createElement("span",{className:"".concat(m,"-inner-unchecked")},v)))});p.displayName="Switch";var u=e.i(121872),m=e.i(242064),g=e.i(937328),f=e.i(517455);e.i(296059);var h=e.i(915654);e.i(262370);var _=e.i(135551),b=e.i(183293),I=e.i(246422),v=e.i(838378);let y=(0,I.genStyleHooks)("Switch",e=>{let t=(0,v.mergeToken)(e,{switchDuration:e.motionDurationMid,switchColor:e.colorPrimary,switchDisabledOpacity:e.opacityLoading,switchLoadingIconSize:e.calc(e.fontSizeIcon).mul(.75).equal(),switchLoadingIconColor:`rgba(0, 0, 0, ${e.opacityLoading})`,switchHandleActiveInset:"-30%"});return[(e=>{let{componentCls:t,trackHeight:a,trackMinWidth:i}=e;return{[t]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,b.resetComponent)(e)),{position:"relative",display:"inline-block",boxSizing:"border-box",minWidth:i,height:a,lineHeight:(0,h.unit)(a),verticalAlign:"middle",background:e.colorTextQuaternary,border:"0",borderRadius:100,cursor:"pointer",transition:`all ${e.motionDurationMid}`,userSelect:"none",[`&:hover:not(${t}-disabled)`]:{background:e.colorTextTertiary}}),(0,b.genFocusStyle)(e)),{[`&${t}-checked`]:{background:e.switchColor,[`&:hover:not(${t}-disabled)`]:{background:e.colorPrimaryHover}},[`&${t}-loading, &${t}-disabled`]:{cursor:"not-allowed",opacity:e.switchDisabledOpacity,"*":{boxShadow:"none",cursor:"not-allowed"}},[`&${t}-rtl`]:{direction:"rtl"}})}})(t),(e=>{let{componentCls:t,trackHeight:a,trackPadding:i,innerMinMargin:n,innerMaxMargin:r,handleSize:o,calc:l}=e,s=`${t}-inner`,c=(0,h.unit)(l(o).add(l(i).mul(2)).equal()),d=(0,h.unit)(l(r).mul(2).equal());return{[t]:{[s]:{display:"block",overflow:"hidden",borderRadius:100,height:"100%",paddingInlineStart:r,paddingInlineEnd:n,transition:`padding-inline-start ${e.switchDuration} ease-in-out, padding-inline-end ${e.switchDuration} ease-in-out`,[`${s}-checked, ${s}-unchecked`]:{display:"block",color:e.colorTextLightSolid,fontSize:e.fontSizeSM,transition:`margin-inline-start ${e.switchDuration} ease-in-out, margin-inline-end ${e.switchDuration} ease-in-out`,pointerEvents:"none",minHeight:a},[`${s}-checked`]:{marginInlineStart:`calc(-100% + ${c} - ${d})`,marginInlineEnd:`calc(100% - ${c} + ${d})`},[`${s}-unchecked`]:{marginTop:l(a).mul(-1).equal(),marginInlineStart:0,marginInlineEnd:0}},[`&${t}-checked ${s}`]:{paddingInlineStart:n,paddingInlineEnd:r,[`${s}-checked`]:{marginInlineStart:0,marginInlineEnd:0},[`${s}-unchecked`]:{marginInlineStart:`calc(100% - ${c} + ${d})`,marginInlineEnd:`calc(-100% + ${c} - ${d})`}},[`&:not(${t}-disabled):active`]:{[`&:not(${t}-checked) ${s}`]:{[`${s}-unchecked`]:{marginInlineStart:l(i).mul(2).equal(),marginInlineEnd:l(i).mul(-1).mul(2).equal()}},[`&${t}-checked ${s}`]:{[`${s}-checked`]:{marginInlineStart:l(i).mul(-1).mul(2).equal(),marginInlineEnd:l(i).mul(2).equal()}}}}}})(t),(e=>{let{componentCls:t,trackPadding:a,handleBg:i,handleShadow:n,handleSize:r,calc:o}=e,l=`${t}-handle`;return{[t]:{[l]:{position:"absolute",top:a,insetInlineStart:a,width:r,height:r,transition:`all ${e.switchDuration} ease-in-out`,"&::before":{position:"absolute",top:0,insetInlineEnd:0,bottom:0,insetInlineStart:0,backgroundColor:i,borderRadius:o(r).div(2).equal(),boxShadow:n,transition:`all ${e.switchDuration} ease-in-out`,content:'""'}},[`&${t}-checked ${l}`]:{insetInlineStart:`calc(100% - ${(0,h.unit)(o(r).add(a).equal())})`},[`&:not(${t}-disabled):active`]:{[`${l}::before`]:{insetInlineEnd:e.switchHandleActiveInset,insetInlineStart:0},[`&${t}-checked ${l}::before`]:{insetInlineEnd:0,insetInlineStart:e.switchHandleActiveInset}}}}})(t),(e=>{let{componentCls:t,handleSize:a,calc:i}=e;return{[t]:{[`${t}-loading-icon${e.iconCls}`]:{position:"relative",top:i(i(a).sub(e.fontSize)).div(2).equal(),color:e.switchLoadingIconColor,verticalAlign:"top"},[`&${t}-checked ${t}-loading-icon`]:{color:e.switchColor}}}})(t),(e=>{let{componentCls:t,trackHeightSM:a,trackPadding:i,trackMinWidthSM:n,innerMinMarginSM:r,innerMaxMarginSM:o,handleSizeSM:l,calc:s}=e,c=`${t}-inner`,d=(0,h.unit)(s(l).add(s(i).mul(2)).equal()),p=(0,h.unit)(s(o).mul(2).equal());return{[t]:{[`&${t}-small`]:{minWidth:n,height:a,lineHeight:(0,h.unit)(a),[`${t}-inner`]:{paddingInlineStart:o,paddingInlineEnd:r,[`${c}-checked, ${c}-unchecked`]:{minHeight:a},[`${c}-checked`]:{marginInlineStart:`calc(-100% + ${d} - ${p})`,marginInlineEnd:`calc(100% - ${d} + ${p})`},[`${c}-unchecked`]:{marginTop:s(a).mul(-1).equal(),marginInlineStart:0,marginInlineEnd:0}},[`${t}-handle`]:{width:l,height:l},[`${t}-loading-icon`]:{top:s(s(l).sub(e.switchLoadingIconSize)).div(2).equal(),fontSize:e.switchLoadingIconSize},[`&${t}-checked`]:{[`${t}-inner`]:{paddingInlineStart:r,paddingInlineEnd:o,[`${c}-checked`]:{marginInlineStart:0,marginInlineEnd:0},[`${c}-unchecked`]:{marginInlineStart:`calc(100% - ${d} + ${p})`,marginInlineEnd:`calc(-100% + ${d} - ${p})`}},[`${t}-handle`]:{insetInlineStart:`calc(100% - ${(0,h.unit)(s(l).add(i).equal())})`}},[`&:not(${t}-disabled):active`]:{[`&:not(${t}-checked) ${c}`]:{[`${c}-unchecked`]:{marginInlineStart:s(e.marginXXS).div(2).equal(),marginInlineEnd:s(e.marginXXS).mul(-1).div(2).equal()}},[`&${t}-checked ${c}`]:{[`${c}-checked`]:{marginInlineStart:s(e.marginXXS).mul(-1).div(2).equal(),marginInlineEnd:s(e.marginXXS).div(2).equal()}}}}}}})(t)]},e=>{let{fontSize:t,lineHeight:a,controlHeight:i,colorWhite:n}=e,r=t*a,o=i/2,l=r-4,s=o-4;return{trackHeight:r,trackHeightSM:o,trackMinWidth:2*l+8,trackMinWidthSM:2*s+4,trackPadding:2,handleBg:n,handleSize:l,handleSizeSM:s,handleShadow:`0 2px 4px 0 ${new _.FastColor("#00230b").setA(.2).toRgbString()}`,innerMinMargin:l/2,innerMaxMargin:l+2+4,innerMinMarginSM:s/2,innerMaxMarginSM:s+2+4}});var w=function(e,t){var a={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(a[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,i=Object.getOwnPropertySymbols(e);n<i.length;n++)0>t.indexOf(i[n])&&Object.prototype.propertyIsEnumerable.call(e,i[n])&&(a[i[n]]=e[i[n]]);return a};let x=t.forwardRef((e,n)=>{let{prefixCls:r,size:o,disabled:l,loading:c,className:d,rootClassName:h,style:_,checked:b,value:I,defaultChecked:v,defaultValue:x,onChange:S}=e,$=w(e,["prefixCls","size","disabled","loading","className","rootClassName","style","checked","value","defaultChecked","defaultValue","onChange"]),[A,k]=(0,s.default)(!1,{value:null!=b?b:I,defaultValue:null!=v?v:x}),{getPrefixCls:E,direction:C,switch:O}=t.useContext(m.ConfigContext),j=t.useContext(g.default),M=(null!=l?l:j)||c,T=E("switch",r),N=t.createElement("div",{className:`${T}-handle`},c&&t.createElement(a.default,{className:`${T}-loading-icon`})),[L,z,R]=y(T),P=(0,f.default)(o),D=(0,i.default)(null==O?void 0:O.className,{[`${T}-small`]:"small"===P,[`${T}-loading`]:c,[`${T}-rtl`]:"rtl"===C},d,h,z,R),G=Object.assign(Object.assign({},null==O?void 0:O.style),_);return L(t.createElement(u.default,{component:"Switch",disabled:M},t.createElement(p,Object.assign({},$,{checked:A,onChange:(...e)=>{k(e[0]),null==S||S.apply(void 0,e)},prefixCls:T,className:D,style:G,disabled:M,ref:n,loadingIcon:N}))))});x.__ANT_SWITCH=!0,e.s(["Switch",0,x],790848)},38243,908286,e=>{"use strict";e.i(247167);var t=e.i(271645),a=e.i(343794),i=e.i(876556);function n(e){return["small","middle","large"].includes(e)}function r(e){return!!e&&"number"==typeof e&&!Number.isNaN(e)}e.s(["isPresetSize",()=>n,"isValidGapNumber",()=>r],908286);var o=e.i(242064),l=e.i(249616),s=e.i(372409),c=e.i(246422);let d=(0,c.genStyleHooks)(["Space","Addon"],e=>[(e=>{let{componentCls:t,borderRadius:a,paddingSM:i,colorBorder:n,paddingXS:r,fontSizeLG:o,fontSizeSM:l,borderRadiusLG:c,borderRadiusSM:d,colorBgContainerDisabled:p,lineWidth:u}=e;return{[t]:[{display:"inline-flex",alignItems:"center",gap:0,paddingInline:i,margin:0,background:p,borderWidth:u,borderStyle:"solid",borderColor:n,borderRadius:a,"&-large":{fontSize:o,borderRadius:c},"&-small":{paddingInline:r,borderRadius:d,fontSize:l},"&-compact-last-item":{borderEndStartRadius:0,borderStartStartRadius:0},"&-compact-first-item":{borderEndEndRadius:0,borderStartEndRadius:0},"&-compact-item:not(:first-child):not(:last-child)":{borderRadius:0},"&-compact-item:not(:last-child)":{borderInlineEndWidth:0}},(0,s.genCompactItemStyle)(e,{focus:!1})]}})(e)]);var p=function(e,t){var a={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(a[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,i=Object.getOwnPropertySymbols(e);n<i.length;n++)0>t.indexOf(i[n])&&Object.prototype.propertyIsEnumerable.call(e,i[n])&&(a[i[n]]=e[i[n]]);return a};let u=t.default.forwardRef((e,i)=>{let{className:n,children:r,style:s,prefixCls:c}=e,u=p(e,["className","children","style","prefixCls"]),{getPrefixCls:m,direction:g}=t.default.useContext(o.ConfigContext),f=m("space-addon",c),[h,_,b]=d(f),{compactItemClassnames:I,compactSize:v}=(0,l.useCompactItemContext)(f,g),y=(0,a.default)(f,_,I,b,{[`${f}-${v}`]:v},n);return h(t.default.createElement("div",Object.assign({ref:i,className:y,style:s},u),r))}),m=t.default.createContext({latestIndex:0}),g=m.Provider,f=({className:e,index:a,children:i,split:n,style:r})=>{let{latestIndex:o}=t.useContext(m);return null==i?null:t.createElement(t.Fragment,null,t.createElement("div",{className:e,style:r},i),a<o&&n&&t.createElement("span",{className:`${e}-split`},n))};var h=e.i(838378);let _=(0,c.genStyleHooks)("Space",e=>{let t=(0,h.mergeToken)(e,{spaceGapSmallSize:e.paddingXS,spaceGapMiddleSize:e.padding,spaceGapLargeSize:e.paddingLG});return[(e=>{let{componentCls:t,antCls:a}=e;return{[t]:{display:"inline-flex","&-rtl":{direction:"rtl"},"&-vertical":{flexDirection:"column"},"&-align":{flexDirection:"column","&-center":{alignItems:"center"},"&-start":{alignItems:"flex-start"},"&-end":{alignItems:"flex-end"},"&-baseline":{alignItems:"baseline"}},[`${t}-item:empty`]:{display:"none"},[`${t}-item > ${a}-badge-not-a-wrapper:only-child`]:{display:"block"}}}})(t),(e=>{let{componentCls:t}=e;return{[t]:{"&-gap-row-small":{rowGap:e.spaceGapSmallSize},"&-gap-row-middle":{rowGap:e.spaceGapMiddleSize},"&-gap-row-large":{rowGap:e.spaceGapLargeSize},"&-gap-col-small":{columnGap:e.spaceGapSmallSize},"&-gap-col-middle":{columnGap:e.spaceGapMiddleSize},"&-gap-col-large":{columnGap:e.spaceGapLargeSize}}}})(t)]},()=>({}),{resetStyle:!1});var b=function(e,t){var a={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(a[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,i=Object.getOwnPropertySymbols(e);n<i.length;n++)0>t.indexOf(i[n])&&Object.prototype.propertyIsEnumerable.call(e,i[n])&&(a[i[n]]=e[i[n]]);return a};let I=t.forwardRef((e,l)=>{var s;let{getPrefixCls:c,direction:d,size:p,className:u,style:m,classNames:h,styles:I}=(0,o.useComponentConfig)("space"),{size:v=null!=p?p:"small",align:y,className:w,rootClassName:x,children:S,direction:$="horizontal",prefixCls:A,split:k,style:E,wrap:C=!1,classNames:O,styles:j}=e,M=b(e,["size","align","className","rootClassName","children","direction","prefixCls","split","style","wrap","classNames","styles"]),[T,N]=Array.isArray(v)?v:[v,v],L=n(N),z=n(T),R=r(N),P=r(T),D=(0,i.default)(S,{keepEmpty:!0}),G=void 0===y&&"horizontal"===$?"center":y,H=c("space",A),[V,q,F]=_(H),B=(0,a.default)(H,u,q,`${H}-${$}`,{[`${H}-rtl`]:"rtl"===d,[`${H}-align-${G}`]:G,[`${H}-gap-row-${N}`]:L,[`${H}-gap-col-${T}`]:z},w,x,F),U=(0,a.default)(`${H}-item`,null!=(s=null==O?void 0:O.item)?s:h.item),W=Object.assign(Object.assign({},I.item),null==j?void 0:j.item),X=D.map((e,a)=>{let i=(null==e?void 0:e.key)||`${U}-${a}`;return t.createElement(f,{className:U,key:i,index:a,split:k,style:W},e)}),Y=t.useMemo(()=>({latestIndex:D.reduce((e,t,a)=>null!=t?a:e,0)}),[D]);if(0===D.length)return null;let J={};return C&&(J.flexWrap="wrap"),!z&&P&&(J.columnGap=T),!L&&R&&(J.rowGap=N),V(t.createElement("div",Object.assign({ref:l,className:B,style:Object.assign(Object.assign(Object.assign({},J),m),E)},M),t.createElement(g,{value:Y},X)))});I.Compact=l.default,I.Addon=u,e.s(["default",0,I],38243)},770914,e=>{"use strict";var t=e.i(38243);e.s(["Space",()=>t.default])},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},771674,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["UserOutlined",0,r],771674)},948401,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M928 160H96c-17.7 0-32 14.3-32 32v640c0 17.7 14.3 32 32 32h832c17.7 0 32-14.3 32-32V192c0-17.7-14.3-32-32-32zm-40 110.8V792H136V270.8l-27.6-21.5 39.3-50.5 42.8 33.3h643.1l42.8-33.3 39.3 50.5-27.7 21.5zM833.6 232L512 482 190.4 232l-42.8-33.3-39.3 50.5 27.6 21.5 341.6 265.6a55.99 55.99 0 0068.7 0L888 270.8l27.6-21.5-39.3-50.5-42.7 33.2z"}}]},name:"mail",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["MailOutlined",0,r],948401)},62478,e=>{"use strict";var t=e.i(764205);let a=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,a])},818581,(e,t,a)=>{"use strict";Object.defineProperty(a,"__esModule",{value:!0}),Object.defineProperty(a,"useMergedRef",{enumerable:!0,get:function(){return n}});let i=e.r(271645);function n(e,t){let a=(0,i.useRef)(null),n=(0,i.useRef)(null);return(0,i.useCallback)(i=>{if(null===i){let e=a.current;e&&(a.current=null,e());let t=n.current;t&&(n.current=null,t())}else e&&(a.current=r(e,i)),t&&(n.current=r(t,i))},[e,t])}function r(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let a=e(t);return"function"==typeof a?a:()=>e(null)}}("function"==typeof a.default||"object"==typeof a.default&&null!==a.default)&&void 0===a.default.__esModule&&(Object.defineProperty(a.default,"__esModule",{value:!0}),Object.assign(a.default,a),t.exports=a.default)},56456,e=>{"use strict";var t=e.i(739295);e.s(["LoadingOutlined",()=>t.default])},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["SafetyOutlined",0,r],602073)},190272,785913,e=>{"use strict";var t,a,i=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),n=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a.REALTIME="realtime",a);let r={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>n,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(i).includes(e)){let t=r[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:i,apiKey:r,inputMessage:o,chatHistory:l,selectedTags:s,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:p,selectedMCPServers:u,mcpServers:m,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:_,selectedSdk:b,proxySettings:I}=e,v="session"===a?i:r,y=window.location.origin,w=I?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?y=w:I?.PROXY_BASE_URL&&(y=I.PROXY_BASE_URL);let x=o||"Your prompt here",S=x.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),$=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),A={};s.length>0&&(A.tags=s),c.length>0&&(A.vector_stores=c),d.length>0&&(A.guardrails=d),p.length>0&&(A.policies=p);let k=_||"your-model-name",E="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"
)`;switch(h){case n.CHAT:{let e=Object.keys(A).length>0,a="";if(e){let e=JSON.stringify({metadata:A},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let i=$.length>0?$:[{role:"user",content:x}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(i,null,4)}${a}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${k}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${S}"
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
`;break}case n.RESPONSES:{let e=Object.keys(A).length>0,a="";if(e){let e=JSON.stringify({metadata:A},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let i=$.length>0?$:[{role:"user",content:x}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(i,null,4)}${a}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${k}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${S}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${a}
# )
# print(response_with_file.output_text)
`;break}case n.IMAGE:t="azure"===b?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${k}",
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${k}",
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
`;break;case n.IMAGE_EDITS:t="azure"===b?`
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${k}",
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${k}",
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
`;break;case n.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${o||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case n.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${o?`,
	prompt="${o.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case n.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${o||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${o||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${E}
${t}`}],190272)},115571,371401,e=>{"use strict";let t="local-storage-change";function a(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function i(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function n(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function r(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>a,"getLocalStorageItem",()=>i,"removeLocalStorageItem",()=>r,"setLocalStorageItem",()=>n],115571);var o=e.i(271645);function l(e){let a=t=>{"disableUsageIndicator"===t.key&&e()},i=t=>{let{key:a}=t.detail;"disableUsageIndicator"===a&&e()};return window.addEventListener("storage",a),window.addEventListener(t,i),()=>{window.removeEventListener("storage",a),window.removeEventListener(t,i)}}function s(){return"true"===i("disableUsageIndicator")}function c(){return(0,o.useSyncExternalStore)(l,s)}e.s(["useDisableUsageIndicator",()=>c],371401)},275144,e=>{"use strict";var t=e.i(843476),a=e.i(271645),i=e.i(764205);let n=(0,a.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:r})=>{let[o,l]=(0,a.useState)(null),[s,c]=(0,a.useState)(null);return(0,a.useEffect)(()=>{(async()=>{try{let e=(0,i.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",a=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(a.ok){let e=await a.json();e.values?.logo_url&&l(e.values.logo_url),e.values?.favicon_url&&c(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,a.useEffect)(()=>{if(s){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=s});else{let e=document.createElement("link");e.rel="icon",e.href=s,document.head.appendChild(e)}}},[s]),(0,t.jsx)(n.Provider,{value:{logoUrl:o,setLogoUrl:l,faviconUrl:s,setFaviconUrl:c},children:e})},"useTheme",0,()=>{let e=(0,a.useContext)(n);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},434626,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,a],434626)},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["CrownOutlined",0,r],100486)},94629,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,a],94629)},991124,e=>{"use strict";let t=(0,e.i(475254).default)("copy",[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]]);e.s(["default",()=>t])},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},798496,e=>{"use strict";var t=e.i(843476),a=e.i(152990),i=e.i(682830),n=e.i(271645),r=e.i(269200),o=e.i(427612),l=e.i(64848),s=e.i(942232),c=e.i(496020),d=e.i(977572),p=e.i(94629),u=e.i(360820),m=e.i(871943);function g({data:e=[],columns:g,isLoading:f=!1,defaultSorting:h=[],pagination:_,onPaginationChange:b,enablePagination:I=!1}){let[v,y]=n.default.useState(h),[w]=n.default.useState("onChange"),[x,S]=n.default.useState({}),[$,A]=n.default.useState({}),k=(0,a.useReactTable)({data:e,columns:g,state:{sorting:v,columnSizing:x,columnVisibility:$,...I&&_?{pagination:_}:{}},columnResizeMode:w,onSortingChange:y,onColumnSizingChange:S,onColumnVisibilityChange:A,...I&&b?{onPaginationChange:b}:{},getCoreRowModel:(0,i.getCoreRowModel)(),getSortedRowModel:(0,i.getSortedRowModel)(),...I?{getPaginationRowModel:(0,i.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(r.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:k.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(o.TableHead,{children:k.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(l.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,a.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(u.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(m.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(p.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(s.TableBody,{children:f?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):k.getRowModel().rows.length>0?k.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,a.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>g])}]);