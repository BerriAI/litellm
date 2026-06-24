(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,906579,100486,e=>{"use strict";e.i(247167);var t=e.i(271645),a=e.i(343794),o=e.i(361275),i=e.i(702779),n=e.i(763731),r=e.i(242064);e.i(296059);var l=e.i(915654),s=e.i(694758),c=e.i(183293),p=e.i(403541),u=e.i(246422),m=e.i(838378);let d=new s.Keyframes("antStatusProcessing",{"0%":{transform:"scale(0.8)",opacity:.5},"100%":{transform:"scale(2.4)",opacity:0}}),g=new s.Keyframes("antZoomBadgeIn",{"0%":{transform:"scale(0) translate(50%, -50%)",opacity:0},"100%":{transform:"scale(1) translate(50%, -50%)"}}),f=new s.Keyframes("antZoomBadgeOut",{"0%":{transform:"scale(1) translate(50%, -50%)"},"100%":{transform:"scale(0) translate(50%, -50%)",opacity:0}}),_=new s.Keyframes("antNoWrapperZoomBadgeIn",{"0%":{transform:"scale(0)",opacity:0},"100%":{transform:"scale(1)"}}),b=new s.Keyframes("antNoWrapperZoomBadgeOut",{"0%":{transform:"scale(1)"},"100%":{transform:"scale(0)",opacity:0}}),h=new s.Keyframes("antBadgeLoadingCircle",{"0%":{transformOrigin:"50%"},"100%":{transform:"translate(50%, -50%) rotate(360deg)",transformOrigin:"50%"}}),A=e=>{let{fontHeight:t,lineWidth:a,marginXS:o,colorBorderBg:i}=e,n=e.colorTextLightSolid,r=e.colorError,l=e.colorErrorHover;return(0,m.mergeToken)(e,{badgeFontHeight:t,badgeShadowSize:a,badgeTextColor:n,badgeColor:r,badgeColorHover:l,badgeShadowColor:i,badgeProcessingDuration:"1.2s",badgeRibbonOffset:o,badgeRibbonCornerTransform:"scaleY(0.75)",badgeRibbonCornerFilter:"brightness(75%)"})},v=e=>{let{fontSize:t,lineHeight:a,fontSizeSM:o,lineWidth:i}=e;return{indicatorZIndex:"auto",indicatorHeight:Math.round(t*a)-2*i,indicatorHeightSM:t,dotSize:o/2,textFontSize:o,textFontSizeSM:o,textFontWeight:"normal",statusSize:o/2}},I=(0,u.genStyleHooks)("Badge",e=>(e=>{let{componentCls:t,iconCls:a,antCls:o,badgeShadowSize:i,textFontSize:n,textFontSizeSM:r,statusSize:s,dotSize:u,textFontWeight:m,indicatorHeight:A,indicatorHeightSM:v,marginXS:I,calc:O}=e,E=`${o}-scroll-number`,y=(0,p.genPresetColor)(e,(e,{darkColor:a})=>({[`&${t} ${t}-color-${e}`]:{background:a,[`&:not(${t}-count)`]:{color:a},"a:hover &":{background:a}}}));return{[t]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,c.resetComponent)(e)),{position:"relative",display:"inline-block",width:"fit-content",lineHeight:1,[`${t}-count`]:{display:"inline-flex",justifyContent:"center",zIndex:e.indicatorZIndex,minWidth:A,height:A,color:e.badgeTextColor,fontWeight:m,fontSize:n,lineHeight:(0,l.unit)(A),whiteSpace:"nowrap",textAlign:"center",background:e.badgeColor,borderRadius:O(A).div(2).equal(),boxShadow:`0 0 0 ${(0,l.unit)(i)} ${e.badgeShadowColor}`,transition:`background ${e.motionDurationMid}`,a:{color:e.badgeTextColor},"a:hover":{color:e.badgeTextColor},"a:hover &":{background:e.badgeColorHover}},[`${t}-count-sm`]:{minWidth:v,height:v,fontSize:r,lineHeight:(0,l.unit)(v),borderRadius:O(v).div(2).equal()},[`${t}-multiple-words`]:{padding:`0 ${(0,l.unit)(e.paddingXS)}`,bdi:{unicodeBidi:"plaintext"}},[`${t}-dot`]:{zIndex:e.indicatorZIndex,width:u,minWidth:u,height:u,background:e.badgeColor,borderRadius:"100%",boxShadow:`0 0 0 ${(0,l.unit)(i)} ${e.badgeShadowColor}`},[`${t}-count, ${t}-dot, ${E}-custom-component`]:{position:"absolute",top:0,insetInlineEnd:0,transform:"translate(50%, -50%)",transformOrigin:"100% 0%",[`&${a}-spin`]:{animationName:h,animationDuration:"1s",animationIterationCount:"infinite",animationTimingFunction:"linear"}},[`&${t}-status`]:{lineHeight:"inherit",verticalAlign:"baseline",[`${t}-status-dot`]:{position:"relative",top:-1,display:"inline-block",width:s,height:s,verticalAlign:"middle",borderRadius:"50%"},[`${t}-status-success`]:{backgroundColor:e.colorSuccess},[`${t}-status-processing`]:{overflow:"visible",color:e.colorInfo,backgroundColor:e.colorInfo,borderColor:"currentcolor","&::after":{position:"absolute",top:0,insetInlineStart:0,width:"100%",height:"100%",borderWidth:i,borderStyle:"solid",borderColor:"inherit",borderRadius:"50%",animationName:d,animationDuration:e.badgeProcessingDuration,animationIterationCount:"infinite",animationTimingFunction:"ease-in-out",content:'""'}},[`${t}-status-default`]:{backgroundColor:e.colorTextPlaceholder},[`${t}-status-error`]:{backgroundColor:e.colorError},[`${t}-status-warning`]:{backgroundColor:e.colorWarning},[`${t}-status-text`]:{marginInlineStart:I,color:e.colorText,fontSize:e.fontSize}}}),y),{[`${t}-zoom-appear, ${t}-zoom-enter`]:{animationName:g,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack,animationFillMode:"both"},[`${t}-zoom-leave`]:{animationName:f,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack,animationFillMode:"both"},[`&${t}-not-a-wrapper`]:{[`${t}-zoom-appear, ${t}-zoom-enter`]:{animationName:_,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack},[`${t}-zoom-leave`]:{animationName:b,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack},[`&:not(${t}-status)`]:{verticalAlign:"middle"},[`${E}-custom-component, ${t}-count`]:{transform:"none"},[`${E}-custom-component, ${E}`]:{position:"relative",top:"auto",display:"block",transformOrigin:"50% 50%"}},[E]:{overflow:"hidden",transition:`all ${e.motionDurationMid} ${e.motionEaseOutBack}`,[`${E}-only`]:{position:"relative",display:"inline-block",height:A,transition:`all ${e.motionDurationSlow} ${e.motionEaseOutBack}`,WebkitTransformStyle:"preserve-3d",WebkitBackfaceVisibility:"hidden",[`> p${E}-only-unit`]:{height:A,margin:0,WebkitTransformStyle:"preserve-3d",WebkitBackfaceVisibility:"hidden"}},[`${E}-symbol`]:{verticalAlign:"top"}},"&-rtl":{direction:"rtl",[`${t}-count, ${t}-dot, ${E}-custom-component`]:{transform:"translate(-50%, -50%)"}}})}})(A(e)),v),O=(0,u.genStyleHooks)(["Badge","Ribbon"],e=>(e=>{let{antCls:t,badgeFontHeight:a,marginXS:o,badgeRibbonOffset:i,calc:n}=e,r=`${t}-ribbon`,s=`${t}-ribbon-wrapper`,u=(0,p.genPresetColor)(e,(e,{darkColor:t})=>({[`&${r}-color-${e}`]:{background:t,color:t}}));return{[s]:{position:"relative"},[r]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,c.resetComponent)(e)),{position:"absolute",top:o,padding:`0 ${(0,l.unit)(e.paddingXS)}`,color:e.colorPrimary,lineHeight:(0,l.unit)(a),whiteSpace:"nowrap",backgroundColor:e.colorPrimary,borderRadius:e.borderRadiusSM,[`${r}-text`]:{color:e.badgeTextColor},[`${r}-corner`]:{position:"absolute",top:"100%",width:i,height:i,color:"currentcolor",border:`${(0,l.unit)(n(i).div(2).equal())} solid`,transform:e.badgeRibbonCornerTransform,transformOrigin:"top",filter:e.badgeRibbonCornerFilter}}),u),{[`&${r}-placement-end`]:{insetInlineEnd:n(i).mul(-1).equal(),borderEndEndRadius:0,[`${r}-corner`]:{insetInlineEnd:0,borderInlineEndColor:"transparent",borderBlockEndColor:"transparent"}},[`&${r}-placement-start`]:{insetInlineStart:n(i).mul(-1).equal(),borderEndStartRadius:0,[`${r}-corner`]:{insetInlineStart:0,borderBlockEndColor:"transparent",borderInlineStartColor:"transparent"}},"&-rtl":{direction:"rtl"}})}})(A(e)),v),E=e=>{let o,{prefixCls:i,value:n,current:r,offset:l=0}=e;return l&&(o={position:"absolute",top:`${l}00%`,left:0}),t.createElement("span",{style:o,className:(0,a.default)(`${i}-only-unit`,{current:r})},n)},y=e=>{let a,o,{prefixCls:i,count:n,value:r}=e,l=Number(r),s=Math.abs(n),[c,p]=t.useState(l),[u,m]=t.useState(s),d=()=>{p(l),m(s)};if(t.useEffect(()=>{let e=setTimeout(d,1e3);return()=>clearTimeout(e)},[l]),c===l||Number.isNaN(l)||Number.isNaN(c))a=[t.createElement(E,Object.assign({},e,{key:l,current:!0}))],o={transition:"none"};else{a=[];let i=l+10,n=[];for(let e=l;e<=i;e+=1)n.push(e);let r=u<s?1:-1,p=n.findIndex(e=>e%10===c);a=(r<0?n.slice(0,p+1):n.slice(p)).map((a,o)=>t.createElement(E,Object.assign({},e,{key:a,value:a%10,offset:r<0?o-p:o,current:o===p}))),o={transform:`translateY(${-function(e,t,a){let o=e,i=0;for(;(o+10)%10!==t;)o+=a,i+=a;return i}(c,l,r)}00%)`}}return t.createElement("span",{className:`${i}-only`,style:o,onTransitionEnd:d},a)};var x=function(e,t){var a={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(a[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,o=Object.getOwnPropertySymbols(e);i<o.length;i++)0>t.indexOf(o[i])&&Object.prototype.propertyIsEnumerable.call(e,o[i])&&(a[o[i]]=e[o[i]]);return a};let $=t.forwardRef((e,o)=>{let{prefixCls:i,count:l,className:s,motionClassName:c,style:p,title:u,show:m,component:d="sup",children:g}=e,f=x(e,["prefixCls","count","className","motionClassName","style","title","show","component","children"]),{getPrefixCls:_}=t.useContext(r.ConfigContext),b=_("scroll-number",i),h=Object.assign(Object.assign({},f),{"data-show":m,style:p,className:(0,a.default)(b,s,c),title:u}),A=l;if(l&&Number(l)%1==0){let e=String(l).split("");A=t.createElement("bdi",null,e.map((a,o)=>t.createElement(y,{prefixCls:b,count:Number(l),value:a,key:e.length-o})))}return((null==p?void 0:p.borderColor)&&(h.style=Object.assign(Object.assign({},p),{boxShadow:`0 0 0 1px ${p.borderColor} inset`})),g)?(0,n.cloneElement)(g,e=>({className:(0,a.default)(`${b}-custom-component`,null==e?void 0:e.className,c)})):t.createElement(d,Object.assign({},h,{ref:o}),A)});var C=function(e,t){var a={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(a[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,o=Object.getOwnPropertySymbols(e);i<o.length;i++)0>t.indexOf(o[i])&&Object.prototype.propertyIsEnumerable.call(e,o[i])&&(a[o[i]]=e[o[i]]);return a};let T=t.forwardRef((e,l)=>{var s,c,p,u,m;let{prefixCls:d,scrollNumberPrefixCls:g,children:f,status:_,text:b,color:h,count:A=null,overflowCount:v=99,dot:O=!1,size:E="default",title:y,offset:x,style:T,className:S,rootClassName:w,classNames:N,styles:R,showZero:k=!1}=e,M=C(e,["prefixCls","scrollNumberPrefixCls","children","status","text","color","count","overflowCount","dot","size","title","offset","style","className","rootClassName","classNames","styles","showZero"]),{getPrefixCls:L,direction:j,badge:P}=t.useContext(r.ConfigContext),D=L("badge",d),[z,H,B]=I(D),F=A>v?`${v}+`:A,G="0"===F||0===F||"0"===b||0===b,V=null===A||G&&!k,U=(null!=_||null!=h)&&V,W=null!=_||!G,X=O&&!G,K=X?"":F,Y=(0,t.useMemo)(()=>((null==K||""===K)&&(null==b||""===b)||G&&!k)&&!X,[K,G,k,X,b]),Z=(0,t.useRef)(A);Y||(Z.current=A);let q=Z.current,J=(0,t.useRef)(K);Y||(J.current=K);let Q=J.current,ee=(0,t.useRef)(X);Y||(ee.current=X);let et=(0,t.useMemo)(()=>{if(!x)return Object.assign(Object.assign({},null==P?void 0:P.style),T);let e={marginTop:x[1]};return"rtl"===j?e.left=Number.parseInt(x[0],10):e.right=-Number.parseInt(x[0],10),Object.assign(Object.assign(Object.assign({},e),null==P?void 0:P.style),T)},[j,x,T,null==P?void 0:P.style]),ea=null!=y?y:"string"==typeof q||"number"==typeof q?q:void 0,eo=!Y&&(0===b?k:!!b&&!0!==b),ei=eo?t.createElement("span",{className:`${D}-status-text`},b):null,en=q&&"object"==typeof q?(0,n.cloneElement)(q,e=>({style:Object.assign(Object.assign({},et),e.style)})):void 0,er=(0,i.isPresetColor)(h,!1),el=(0,a.default)(null==N?void 0:N.indicator,null==(s=null==P?void 0:P.classNames)?void 0:s.indicator,{[`${D}-status-dot`]:U,[`${D}-status-${_}`]:!!_,[`${D}-color-${h}`]:er}),es={};h&&!er&&(es.color=h,es.background=h);let ec=(0,a.default)(D,{[`${D}-status`]:U,[`${D}-not-a-wrapper`]:!f,[`${D}-rtl`]:"rtl"===j},S,w,null==P?void 0:P.className,null==(c=null==P?void 0:P.classNames)?void 0:c.root,null==N?void 0:N.root,H,B);if(!f&&U&&(b||W||!V)){let e=et.color;return z(t.createElement("span",Object.assign({},M,{className:ec,style:Object.assign(Object.assign(Object.assign({},null==R?void 0:R.root),null==(p=null==P?void 0:P.styles)?void 0:p.root),et)}),t.createElement("span",{className:el,style:Object.assign(Object.assign(Object.assign({},null==R?void 0:R.indicator),null==(u=null==P?void 0:P.styles)?void 0:u.indicator),es)}),eo&&t.createElement("span",{style:{color:e},className:`${D}-status-text`},b)))}return z(t.createElement("span",Object.assign({ref:l},M,{className:ec,style:Object.assign(Object.assign({},null==(m=null==P?void 0:P.styles)?void 0:m.root),null==R?void 0:R.root)}),f,t.createElement(o.default,{visible:!Y,motionName:`${D}-zoom`,motionAppear:!1,motionDeadline:1e3},({className:e})=>{var o,i;let n=L("scroll-number",g),r=ee.current,l=(0,a.default)(null==N?void 0:N.indicator,null==(o=null==P?void 0:P.classNames)?void 0:o.indicator,{[`${D}-dot`]:r,[`${D}-count`]:!r,[`${D}-count-sm`]:"small"===E,[`${D}-multiple-words`]:!r&&Q&&Q.toString().length>1,[`${D}-status-${_}`]:!!_,[`${D}-color-${h}`]:er}),s=Object.assign(Object.assign(Object.assign({},null==R?void 0:R.indicator),null==(i=null==P?void 0:P.styles)?void 0:i.indicator),et);return h&&!er&&((s=s||{}).background=h),t.createElement($,{prefixCls:n,show:!Y,motionClassName:e,className:l,count:Q,title:ea,style:s,key:"scrollNumber"},en)}),ei))});T.Ribbon=e=>{let{className:o,prefixCls:n,style:l,color:s,children:c,text:p,placement:u="end",rootClassName:m}=e,{getPrefixCls:d,direction:g}=t.useContext(r.ConfigContext),f=d("ribbon",n),_=`${f}-wrapper`,[b,h,A]=O(f,_),v=(0,i.isPresetColor)(s,!1),I=(0,a.default)(f,`${f}-placement-${u}`,{[`${f}-rtl`]:"rtl"===g,[`${f}-color-${s}`]:v},o),E={},y={};return s&&!v&&(E.background=s,y.color=s),b(t.createElement("div",{className:(0,a.default)(_,m,h,A)},c,t.createElement("div",{className:(0,a.default)(I,h),style:Object.assign(Object.assign({},E),l)},t.createElement("span",{className:`${f}-text`},p),t.createElement("div",{className:`${f}-corner`,style:y}))))},e.s(["Badge",0,T],906579);var S=e.i(931067);let w={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var N=e.i(9583),R=t.forwardRef(function(e,a){return t.createElement(N.default,(0,S.default)({},e,{ref:a,icon:w}))});e.s(["CrownOutlined",0,R],100486)},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},916925,e=>{"use strict";var t,a=((t={}).A2A_Agent="A2A Agent",t.AI21="Ai21",t.AI21_CHAT="Ai21 Chat",t.AIML="AI/ML API",t.AIOHTTP_OPENAI="Aiohttp Openai",t.Anthropic="Anthropic",t.ANTHROPIC_TEXT="Anthropic Text",t.AssemblyAI="AssemblyAI",t.AUTO_ROUTER="Auto Router",t.Bedrock="Amazon Bedrock",t.BedrockMantle="Amazon Bedrock Mantle",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.AZURE_TEXT="Azure Text",t.BASETEN="Baseten",t.BYTEZ="Bytez",t.Cerebras="Cerebras",t.CLARIFAI="Clarifai",t.CLOUDFLARE="Cloudflare",t.CODESTRAL="Codestral",t.Cohere="Cohere",t.COHERE_CHAT="Cohere Chat",t.COMETAPI="Cometapi",t.COMPACTIFAI="Compactifai",t.Cursor="Cursor",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DATAROBOT="Datarobot",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.DOCKER_MODEL_RUNNER="Docker Model Runner",t.DOTPROMPT="Dotprompt",t.ElevenLabs="ElevenLabs",t.EMPOWER="Empower",t.FalAI="Fal AI",t.FEATHERLESS_AI="Featherless Ai",t.FireworksAI="Fireworks AI",t.FRIENDLIAI="Friendliai",t.GALADRIEL="Galadriel",t.GITHUB_COPILOT="Github Copilot",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.HEROKU="Heroku",t.Hosted_Vllm="vllm",t.HUGGINGFACE="Huggingface",t.HYPERBOLIC="Hyperbolic",t.Infinity="Infinity",t.JinaAI="Jina AI",t.LAMBDA_AI="Lambda Ai",t.LEMONADE="Lemonade",t.LLAMAFILE="Llamafile",t.LM_STUDIO="Lm Studio",t.LLAMA="Meta Llama",t.MARITALK="Maritalk",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.MOONSHOT="Moonshot",t.MORPH="Morph",t.NEBIUS="Nebius",t.NLP_CLOUD="Nlp Cloud",t.NOVITA="Novita",t.NSCALE="Nscale",t.NVIDIA_NIM="Nvidia Nim",t.Ollama="Ollama",t.OLLAMA_CHAT="Ollama Chat",t.OOBABOOGA="Oobabooga",t.OpenAI="OpenAI",t.OPENAI_LIKE="Openai Like",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.OVHCLOUD="Ovhcloud",t.Perplexity="Perplexity",t.PETALS="Petals",t.PG_VECTOR="Pg Vector",t.PREDIBASE="Predibase",t.RECRAFT="Recraft",t.REPLICATE="Replicate",t.RunwayML="RunwayML",t.SAGEMAKER_LEGACY="Sagemaker",t.Sambanova="Sambanova",t.SAP="SAP Generative AI Hub",t.Snowflake="Snowflake",t.TEXT_COMPLETION_CODESTRAL="Text-Completion-Codestral",t.TogetherAI="TogetherAI",t.TOPAZ="Topaz",t.Triton="Triton",t.V0="V0",t.VERCEL_AI_GATEWAY="Vercel Ai Gateway",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VERTEX_AI_BETA="Vertex Ai Beta",t.VLLM="Vllm",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.WANDB="Wandb",t.WATSONX="Watsonx",t.WATSONX_TEXT="Watsonx Text",t.xAI="xAI",t.XINFERENCE="Xinference",t.ZAI="Z.AI (Zhipu AI)",t);let o={A2A_Agent:"a2a_agent",AI21:"ai21",AI21_CHAT:"ai21_chat",AIML:"aiml",AIOHTTP_OPENAI:"aiohttp_openai",Anthropic:"anthropic",ANTHROPIC_TEXT:"anthropic_text",AssemblyAI:"assemblyai",AUTO_ROUTER:"auto_router",Azure:"azure",Azure_AI_Studio:"azure_ai",AZURE_TEXT:"azure_text",BASETEN:"baseten",Bedrock:"bedrock",BedrockMantle:"bedrock_mantle",BYTEZ:"bytez",Cerebras:"cerebras",CLARIFAI:"clarifai",CLOUDFLARE:"cloudflare",CODESTRAL:"codestral",Cohere:"cohere",COHERE_CHAT:"cohere_chat",COMETAPI:"cometapi",COMPACTIFAI:"compactifai",Cursor:"cursor",Dashscope:"dashscope",Databricks:"databricks",DATAROBOT:"datarobot",DeepInfra:"deepinfra",Deepgram:"deepgram",Deepseek:"deepseek",DOCKER_MODEL_RUNNER:"docker_model_runner",DOTPROMPT:"dotprompt",ElevenLabs:"elevenlabs",EMPOWER:"empower",FalAI:"fal_ai",FEATHERLESS_AI:"featherless_ai",FireworksAI:"fireworks_ai",FRIENDLIAI:"friendliai",GALADRIEL:"galadriel",GITHUB_COPILOT:"github_copilot",Google_AI_Studio:"gemini",GradientAI:"gradient_ai",Groq:"groq",HEROKU:"heroku",Hosted_Vllm:"hosted_vllm",HUGGINGFACE:"huggingface",HYPERBOLIC:"hyperbolic",Infinity:"infinity",JinaAI:"jina_ai",LAMBDA_AI:"lambda_ai",LEMONADE:"lemonade",LLAMAFILE:"llamafile",LLAMA:"meta_llama",LM_STUDIO:"lm_studio",MARITALK:"maritalk",MiniMax:"minimax",MistralAI:"mistral",MOONSHOT:"moonshot",MORPH:"morph",NEBIUS:"nebius",NLP_CLOUD:"nlp_cloud",NOVITA:"novita",NSCALE:"nscale",NVIDIA_NIM:"nvidia_nim",Ollama:"ollama",OLLAMA_CHAT:"ollama_chat",OOBABOOGA:"oobabooga",OpenAI:"openai",OPENAI_LIKE:"openai_like",OpenAI_Compatible:"openai",OpenAI_Text:"text-completion-openai",OpenAI_Text_Compatible:"text-completion-openai",Openrouter:"openrouter",Oracle:"oci",OVHCLOUD:"ovhcloud",Perplexity:"perplexity",PETALS:"petals",PG_VECTOR:"pg_vector",PREDIBASE:"predibase",RECRAFT:"recraft",REPLICATE:"replicate",RunwayML:"runwayml",SAGEMAKER_LEGACY:"sagemaker",SageMaker:"sagemaker_chat",Sambanova:"sambanova",SAP:"sap",Snowflake:"snowflake",TEXT_COMPLETION_CODESTRAL:"text-completion-codestral",TogetherAI:"together_ai",TOPAZ:"topaz",Triton:"triton",V0:"v0",VERCEL_AI_GATEWAY:"vercel_ai_gateway",Vertex_AI:"vertex_ai",VERTEX_AI_BETA:"vertex_ai_beta",VLLM:"vllm",VolcEngine:"volcengine",Voyage:"voyage",WANDB:"wandb",WATSONX:"watsonx",WATSONX_TEXT:"watsonx_text",xAI:"xai",XINFERENCE:"xinference",ZAI:"zai"},i="../ui/assets/logos/",n={"A2A Agent":`${i}a2a_agent.png`,Ai21:`${i}ai21.svg`,"Ai21 Chat":`${i}ai21.svg`,"AI/ML API":`${i}aiml_api.svg`,"Aiohttp Openai":`${i}openai_small.svg`,Anthropic:`${i}anthropic.svg`,"Anthropic Text":`${i}anthropic.svg`,AssemblyAI:`${i}assemblyai_small.png`,Azure:`${i}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${i}microsoft_azure.svg`,"Azure Text":`${i}microsoft_azure.svg`,Baseten:`${i}baseten.svg`,"Amazon Bedrock":`${i}bedrock.svg`,"Amazon Bedrock Mantle":`${i}bedrock.svg`,"AWS SageMaker":`${i}bedrock.svg`,Cerebras:`${i}cerebras.svg`,Cloudflare:`${i}cloudflare.svg`,Codestral:`${i}mistral.svg`,Cohere:`${i}cohere.svg`,"Cohere Chat":`${i}cohere.svg`,Cometapi:`${i}cometapi.svg`,Cursor:`${i}cursor.svg`,"Databricks (Qwen API)":`${i}databricks.svg`,Dashscope:`${i}dashscope.svg`,Deepseek:`${i}deepseek.svg`,Deepgram:`${i}deepgram.png`,DeepInfra:`${i}deepinfra.png`,ElevenLabs:`${i}elevenlabs.png`,"Fal AI":`${i}fal_ai.jpg`,"Featherless Ai":`${i}featherless.svg`,"Fireworks AI":`${i}fireworks.svg`,Friendliai:`${i}friendli.svg`,"Github Copilot":`${i}github_copilot.svg`,"Google AI Studio":`${i}google.svg`,GradientAI:`${i}gradientai.svg`,Groq:`${i}groq.svg`,vllm:`${i}vllm.png`,Huggingface:`${i}huggingface.svg`,Hyperbolic:`${i}hyperbolic.svg`,Infinity:`${i}infinity.png`,"Jina AI":`${i}jina.png`,"Lambda Ai":`${i}lambda.svg`,"Lm Studio":`${i}lmstudio.svg`,"Meta Llama":`${i}meta_llama.svg`,MiniMax:`${i}minimax.svg`,"Mistral AI":`${i}mistral.svg`,Moonshot:`${i}moonshot.svg`,Morph:`${i}morph.svg`,Nebius:`${i}nebius.svg`,Novita:`${i}novita.svg`,"Nvidia Nim":`${i}nvidia_nim.svg`,Ollama:`${i}ollama.svg`,"Ollama Chat":`${i}ollama.svg`,Oobabooga:`${i}openai_small.svg`,OpenAI:`${i}openai_small.svg`,"Openai Like":`${i}openai_small.svg`,"OpenAI Text Completion":`${i}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${i}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${i}openai_small.svg`,Openrouter:`${i}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${i}oracle.svg`,Perplexity:`${i}perplexity-ai.svg`,Recraft:`${i}recraft.svg`,Replicate:`${i}replicate.svg`,RunwayML:`${i}runwayml.png`,Sagemaker:`${i}bedrock.svg`,Sambanova:`${i}sambanova.svg`,"SAP Generative AI Hub":`${i}sap.png`,Snowflake:`${i}snowflake.svg`,"Text-Completion-Codestral":`${i}mistral.svg`,TogetherAI:`${i}togetherai.svg`,Topaz:`${i}topaz.svg`,Triton:`${i}nvidia_triton.png`,V0:`${i}v0.svg`,"Vercel Ai Gateway":`${i}vercel.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${i}google.svg`,"Vertex Ai Beta":`${i}google.svg`,Vllm:`${i}vllm.png`,VolcEngine:`${i}volcengine.png`,"Voyage AI":`${i}voyage.webp`,Watsonx:`${i}watsonx.svg`,"Watsonx Text":`${i}watsonx.svg`,xAI:`${i}xai.svg`,Xinference:`${i}xinference.svg`};e.s(["Providers",()=>a,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else if("Cursor"===e)return"cursor/claude-4-sonnet";else if("Z.AI (Zhipu AI)"===e)return"zai/glm-4.5";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:n[e],displayName:e}}let t=Object.keys(o).find(t=>o[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let i=a[t];return{logo:n[i],displayName:i}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let a=o[e];console.log(`Provider mapped to: ${a}`);let i=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let o=t.litellm_provider;(o===a||"string"==typeof o&&(o.startsWith(`${a}_`)||o.startsWith(`${a}-`)))&&i.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&i.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&i.push(e)}))),i},"providerLogoMap",0,n,"provider_map",0,o])},798496,e=>{"use strict";var t=e.i(843476),a=e.i(152990),o=e.i(682830),i=e.i(271645),n=e.i(269200),r=e.i(427612),l=e.i(64848),s=e.i(942232),c=e.i(496020),p=e.i(977572),u=e.i(94629),m=e.i(360820),d=e.i(871943);e.s(["ModelDataTable",0,function({data:e=[],columns:g,isLoading:f=!1,defaultSorting:_=[],pagination:b,onPaginationChange:h,enablePagination:A=!1,onRowClick:v}){let[I,O]=i.default.useState(_),[E]=i.default.useState("onChange"),[y,x]=i.default.useState({}),[$,C]=i.default.useState({}),T=(0,a.useReactTable)({data:e,columns:g,state:{sorting:I,columnSizing:y,columnVisibility:$,...A&&b?{pagination:b}:{}},columnResizeMode:E,onSortingChange:O,onColumnSizingChange:x,onColumnVisibilityChange:C,...A&&h?{onPaginationChange:h}:{},getCoreRowModel:(0,o.getCoreRowModel)(),getSortedRowModel:(0,o.getSortedRowModel)(),...A?{getPaginationRowModel:(0,o.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(n.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:T.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(r.TableHead,{children:T.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(l.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,a.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(m.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(d.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(u.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(s.TableBody,{children:f?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(p.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):T.getRowModel().rows.length>0?T.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{onClick:()=>v?.(e.original),className:v?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(p.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,a.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(p.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}])},928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},44121,186515,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M408 442h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8zm-8 204c0 4.4 3.6 8 8 8h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56zm504-486H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zm0 632H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zM115.4 518.9L271.7 642c5.8 4.6 14.4.5 14.4-6.9V388.9c0-7.4-8.5-11.5-14.4-6.9L115.4 505.1a8.74 8.74 0 000 13.8z"}}]},name:"menu-fold",theme:"outlined"};var i=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(i.default,(0,t.default)({},e,{ref:n,icon:o}))});e.s(["MenuFoldOutlined",0,n],44121);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M408 442h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8zm-8 204c0 4.4 3.6 8 8 8h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56zm504-486H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zm0 632H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zM142.4 642.1L298.7 519a8.84 8.84 0 000-13.9L142.4 381.9c-5.8-4.6-14.4-.5-14.4 6.9v246.3a8.9 8.9 0 0014.4 7z"}}]},name:"menu-unfold",theme:"outlined"};var l=a.forwardRef(function(e,o){return a.createElement(i.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["MenuUnfoldOutlined",0,l],186515)},283713,e=>{"use strict";var t=e.i(271645),a=e.i(764205),o=e.i(612256);let i="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,o.useUIConfig)(),n=e?.is_control_plane??!1,r=e?.workers??[],[l,s]=(0,t.useState)(()=>localStorage.getItem(i));(0,t.useEffect)(()=>{if(!l||0===r.length)return;let e=r.find(e=>e.worker_id===l);e&&(0,a.switchToWorkerUrl)(e.url)},[l,r]);let c=r.find(e=>e.worker_id===l)??null,p=(0,t.useCallback)(e=>{let t=r.find(t=>t.worker_id===e);t&&(s(e),localStorage.setItem(i,e),(0,a.switchToWorkerUrl)(t.url))},[r]);return{isControlPlane:n,workers:r,selectedWorkerId:l,selectedWorker:c,selectWorker:p,disconnectFromWorker:(0,t.useCallback)(()=>{s(null),localStorage.removeItem(i),(0,a.switchToWorkerUrl)(null)},[])}}])},295320,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M704 446H320c-4.4 0-8 3.6-8 8v402c0 4.4 3.6 8 8 8h384c4.4 0 8-3.6 8-8V454c0-4.4-3.6-8-8-8zm-328 64h272v117H376V510zm272 290H376V683h272v117z"}},{tag:"path",attrs:{d:"M424 748a32 32 0 1064 0 32 32 0 10-64 0zm0-178a32 32 0 1064 0 32 32 0 10-64 0z"}},{tag:"path",attrs:{d:"M811.4 368.9C765.6 248 648.9 162 512.2 162S258.8 247.9 213 368.8C126.9 391.5 63.5 470.2 64 563.6 64.6 668 145.6 752.9 247.6 762c4.7.4 8.7-3.3 8.7-8v-60.4c0-4-3-7.4-7-7.9-27-3.4-52.5-15.2-72.1-34.5-24-23.5-37.2-55.1-37.2-88.6 0-28 9.1-54.4 26.2-76.4 16.7-21.4 40.2-36.9 66.1-43.7l37.9-10 13.9-36.7c8.6-22.8 20.6-44.2 35.7-63.5 14.9-19.2 32.6-36 52.4-50 41.1-28.9 89.5-44.2 140-44.2s98.9 15.3 140 44.3c19.9 14 37.5 30.8 52.4 50 15.1 19.3 27.1 40.7 35.7 63.5l13.8 36.6 37.8 10c54.2 14.4 92.1 63.7 92.1 120 0 33.6-13.2 65.1-37.2 88.6-19.5 19.2-44.9 31.1-71.9 34.5-4 .5-6.9 3.9-6.9 7.9V754c0 4.7 4.1 8.4 8.8 8 101.7-9.2 182.5-94 183.2-198.2.6-93.4-62.7-172.1-148.6-194.9z"}}]},name:"cloud-server",theme:"outlined"};var i=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(i.default,(0,t.default)({},e,{ref:n,icon:o}))});e.s(["CloudServerOutlined",0,n],295320)},62478,e=>{"use strict";var t=e.i(764205);let a=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,a])},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var i=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(i.default,(0,t.default)({},e,{ref:n,icon:o}))});e.s(["SafetyOutlined",0,n],602073)},818581,(e,t,a)=>{"use strict";Object.defineProperty(a,"__esModule",{value:!0}),Object.defineProperty(a,"useMergedRef",{enumerable:!0,get:function(){return i}});let o=e.r(271645);function i(e,t){let a=(0,o.useRef)(null),i=(0,o.useRef)(null);return(0,o.useCallback)(o=>{if(null===o){let e=a.current;e&&(a.current=null,e());let t=i.current;t&&(i.current=null,t())}else e&&(a.current=n(e,o)),t&&(i.current=n(t,o))},[e,t])}function n(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let a=e(t);return"function"==typeof a?a:()=>e(null)}}("function"==typeof a.default||"object"==typeof a.default&&null!==a.default)&&void 0===a.default.__esModule&&(Object.defineProperty(a.default,"__esModule",{value:!0}),Object.assign(a.default,a),t.exports=a.default)},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var i=e.i(9583),n=a.forwardRef(function(e,n){return a.createElement(i.default,(0,t.default)({},e,{ref:n,icon:o}))});e.s(["LinkOutlined",0,n],596239)},190272,785913,e=>{"use strict";var t,a,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),i=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a.REALTIME="realtime",a.INTERACTIONS="interactions",a);let n={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>i,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(o).includes(e)){let t=n[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:o,apiKey:n,inputMessage:r,chatHistory:l,selectedTags:s,selectedVectorStores:c,selectedGuardrails:p,selectedPolicies:u,selectedMCPServers:m,mcpServers:d,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:_,selectedModel:b,selectedSdk:h,proxySettings:A}=e,v="session"===a?o:n,I=window.location.origin,O=A?.LITELLM_UI_API_DOC_BASE_URL;O&&O.trim()?I=O:A?.PROXY_BASE_URL&&(I=A.PROXY_BASE_URL);let E=r||"Your prompt here",y=E.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),x=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),$={};s.length>0&&($.tags=s),c.length>0&&($.vector_stores=c),p.length>0&&($.guardrails=p),u.length>0&&($.policies=u);let C=b||"your-model-name",T="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${I}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${I}"
)`;switch(_){case i.CHAT:{let e=Object.keys($).length>0,a="";if(e){let e=JSON.stringify({metadata:$},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let o=x.length>0?x:[{role:"user",content:E}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(o,null,4)}${a}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${C}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${y}"
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
`;break}case i.RESPONSES:{let e=Object.keys($).length>0,a="";if(e){let e=JSON.stringify({metadata:$},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let o=x.length>0?x:[{role:"user",content:E}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(o,null,4)}${a}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${C}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${y}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${a}
# )
# print(response_with_file.output_text)
`;break}case i.IMAGE:t="azure"===h?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${C}",
	prompt="${r}",
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
`;break;case i.IMAGE_EDITS:t="azure"===h?`
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
`;break;case i.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${r||"Your string here"}",
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case i.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${r?`,
	prompt="${r.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case i.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${r||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${r||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${T}
${t}`}],190272)}]);