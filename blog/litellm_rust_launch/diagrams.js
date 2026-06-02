import React from 'react';

const s = {
  fig: {margin: '2.5rem 0', fontFamily: 'inherit'},
  box: {borderRadius: 12, border: '1px solid #e5e7eb', background: '#fff', padding: '2rem 2.5rem'},
  label: {fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#9ca3af', textAlign: 'center', marginBottom: '1.5rem'},
  caption: {textAlign: 'center', fontSize: 12, color: '#9ca3af', marginTop: 12},
  node: (border='#d1d5db', bg='#f9fafb') => ({
    border: `1px solid ${border}`, borderRadius: 6, padding: '8px 20px',
    fontSize: 13, background: bg, display: 'inline-block',
  }),
};

const SmallArrow = ({color='#9ca3af'}) => (
  <svg width="2" height="28" style={{display:'block'}}>
    <line x1="1" y1="0" x2="1" y2="22" stroke={color} strokeWidth="1.5"/>
    <polygon points="1,28 -2,21 4,21" fill={color}/>
  </svg>
);

export function LatencyCompare() {
  const bar = (label, widthPct, valueText, color, textColor='#fff') => (
    <div style={{display:'flex', alignItems:'center', gap:16, marginBottom:14}}>
      <div style={{width:160, fontSize:13, color:'#4b5563', textAlign:'right'}}>{label}</div>
      <div style={{flex:1, background:'#f3f4f6', borderRadius:6, height:32, position:'relative'}}>
        <div style={{width:`${widthPct}%`, height:'100%', background:color, borderRadius:6, display:'flex', alignItems:'center', justifyContent:'flex-end', paddingRight:12}}>
          <span style={{fontSize:12, color:textColor, fontWeight:600}}>{valueText}</span>
        </div>
      </div>
    </div>
  );
  return (
    <figure style={s.fig}>
      <div style={s.box}>
        <p style={s.label}>Gateway overhead per Claude Code call</p>
        {bar('LiteLLM-Rust (target)', 4, '<1ms', '#7c6dff')}
        {bar('Python (typical)', 60, 'ms-scale', '#9ca3af')}
        <div style={{marginTop:16, paddingTop:16, borderTop:'1px solid #e5e7eb', fontSize:12, color:'#6b7280', textAlign:'center'}}>
          Per-call overhead compounds across dozens of tool calls in a single agent run
        </div>
      </div>
      <figcaption style={s.caption}>Sub-millisecond target on the hot path — Python removed from request forwarding</figcaption>
    </figure>
  );
}

export function DropInMigration() {
  return (
    <figure style={s.fig}>
      <div style={s.box}>
        <p style={s.label}>Drop-in migration — same config, same DB</p>
        <div style={{display:'grid', gridTemplateColumns:'1fr auto 1fr', gap:24, alignItems:'center'}}>
          <div style={{border:'1px solid #e5e7eb', borderRadius:8, padding:20, textAlign:'center'}}>
            <p style={{fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.1em', color:'#9ca3af', marginBottom:12}}>Before</p>
            <div style={{...s.node(), marginBottom:8, width:'80%'}}>litellm (Python)</div>
            <div style={{fontSize:11, color:'#6b7280', marginTop:12}}>
              <div>config.yaml</div>
              <div>Postgres DB</div>
              <div>Client SDKs</div>
            </div>
          </div>
          <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:4}}>
            <svg width="40" height="16"><polygon points="0,8 32,8 32,2 40,8 32,14 32,8" fill="#7c6dff"/></svg>
            <span style={{fontSize:10, color:'#7c6dff', fontWeight:600}}>swap binary</span>
          </div>
          <div style={{border:'1px solid #7c6dff', borderRadius:8, padding:20, textAlign:'center', background:'#faf9ff'}}>
            <p style={{fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.1em', color:'#7c6dff', marginBottom:12}}>After</p>
            <div style={{...s.node('#7c6dff','#7c6dff'), color:'#fff', marginBottom:8, width:'80%', fontWeight:600}}>litellm-rust</div>
            <div style={{fontSize:11, color:'#6b7280', marginTop:12}}>
              <div>config.yaml <span style={{color:'#16a34a'}}>(unchanged)</span></div>
              <div>Postgres DB <span style={{color:'#16a34a'}}>(unchanged)</span></div>
              <div>Client SDKs <span style={{color:'#16a34a'}}>(unchanged)</span></div>
            </div>
          </div>
        </div>
      </div>
      <figcaption style={s.caption}>Only the runtime changes — config, DB schema, and client contract are identical</figcaption>
    </figure>
  );
}

export function RemoteAgentsFlow() {
  const pill = (text, dim=false) => (
    <span style={{display:'inline-block', fontSize:11, padding:'3px 10px', borderRadius:999, border:`1px solid ${dim?'#e5e7eb':'#7c6dff'}`, color: dim?'#9ca3af':'#7c6dff', background: dim?'#f9fafb':'#faf9ff', marginRight:6, marginBottom:6}}>{text}</span>
  );
  return (
    <figure style={s.fig}>
      <div style={s.box}>
        <p style={s.label}>Coding-agent runtime in the gateway</p>
        <div style={{display:'flex', flexDirection:'column', alignItems:'center'}}>
          <div style={s.node()}>Trigger (cron · webhook · API)</div>
          <SmallArrow color="#7c6dff"/>
          <div style={{...s.node('#7c6dff','#7c6dff'), color:'#fff', fontWeight:600}}>LiteLLM-Rust Gateway</div>
          <SmallArrow />
          <div style={{...s.node(), textAlign:'center', fontSize:13, padding:'12px 24px'}}>
            <div style={{fontWeight:600, marginBottom:6}}>Sandbox</div>
            <div>{pill('E2B')}{pill('Daytona')}</div>
            <div style={{fontSize:11, color:'#9ca3af', marginTop:4}}>Claude Code runs isolated</div>
          </div>
          <SmallArrow />
          <div style={{...s.node(), textAlign:'center', fontSize:13, padding:'12px 24px'}}>
            <div style={{fontWeight:600, marginBottom:6, color:'#9ca3af'}}>Roadmap</div>
            <div>{pill('durable sessions', true)}{pill('memory', true)}{pill('artifacts', true)}{pill('vault', true)}</div>
          </div>
          <SmallArrow />
          <div style={{...s.node('#111827','#111827'), color:'#fff', fontWeight:600}}>Results streamed back to caller</div>
        </div>
      </div>
      <figcaption style={s.caption}>One runtime: gateway, scheduler, sandbox — proxying LLM calls and running the agents that make them</figcaption>
    </figure>
  );
}
