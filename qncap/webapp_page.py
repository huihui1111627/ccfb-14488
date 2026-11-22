"""可视化页面 HTML(原生 SVG + JS, 无外部依赖, 可离线)。"""

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>不动产登记系统 · 排队网络容量评估</title>
<style>
  :root { --bg:#0f1420; --panel:#182030; --line:#2a3550; --txt:#dfe6f3;
          --muted:#8595b0; --accent:#5b9dff; --warn:#ff8a5b; --ok:#46d39a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  header { padding:16px 24px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; }
  header .sub { color:var(--muted); font-size:13px; }
  .layout { display:grid; grid-template-columns: 1.5fr 1fr; gap:16px;
            padding:16px 24px; }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:12px; padding:16px; }
  .card h2 { font-size:14px; margin:0 0 12px; color:var(--accent);
             font-weight:600; }
  svg { width:100%; height:340px; }
  .node rect { fill:#212d44; stroke:#3a4a6b; stroke-width:1.5; rx:9; }
  .node.bottleneck rect { stroke:var(--warn); stroke-width:2.5; }
  .node text { fill:var(--txt); font-size:12px; }
  .node .util { font-size:16px; font-weight:700; }
  .node .meta { fill:var(--muted); font-size:10px; }
  .edge { stroke:#46577a; stroke-width:1.5; fill:none; marker-end:url(#arrow); }
  .edge-label { fill:var(--muted); font-size:9px; }
  .slider-row { margin-bottom:16px; }
  .slider-row label { display:flex; justify-content:space-between;
            font-size:13px; margin-bottom:6px; }
  .slider-row .val { color:var(--accent); font-weight:700; }
  input[type=range] { width:100%; accent-color:var(--accent); }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th,td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--line); }
  th:first-child,td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:500; }
  .bar { height:7px; background:#2a3550; border-radius:4px; overflow:hidden; }
  .bar > div { height:100%; background:linear-gradient(90deg,#46d39a,#5b9dff); }
  .bar.hot > div { background:linear-gradient(90deg,#ffd166,#ff8a5b); }
  .badge { display:inline-block; padding:1px 7px; border-radius:10px;
           font-size:10px; background:#2a3550; color:var(--muted); }
  .badge.fcfs{background:#26405e;color:#9cc4ff;}
  .badge.ps{background:#3a2e5e;color:#c9a8ff;}
  .badge.is{background:#24443a;color:#86e6c0;}
  .warnbox { background:#3a2320; border:1px solid #6b3a2e; color:#ffb59e;
             border-radius:8px; padding:8px 10px; font-size:12px; margin-top:10px; }
  .okbox { color:var(--ok); font-size:12px; margin-top:8px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:0 24px 24px; }
  .chain { font-size:12px; color:var(--muted); line-height:1.9; }
  .chain b { color:var(--warn); }
  @media (max-width:980px){ .layout{grid-template-columns:1fr;} .grid2{grid-template-columns:1fr;} }
</style>
</head>
<body>
<header>
  <h1>不动产登记系统 · 排队网络容量评估</h1>
  <span class="sub">解析 MVA 实时求解 · 离散事件仿真离线验收 · 拖动滑块观察利用率与响应时间</span>
</header>

<div class="layout">
  <div class="card">
    <h2>排队网络拓扑(节点颜色随利用率变化, 橙框=当前瓶颈)</h2>
    <svg id="net" viewBox="0 0 760 340" preserveAspectRatio="xMidYMid meet">
      <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3"
        orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#46577a"/></marker></defs>
    </svg>
    <div id="chain" class="chain"></div>
  </div>

  <div class="card">
    <h2>负载调节</h2>
    <div class="slider-row">
      <label>查询类闭合用户数 <span class="val" id="v-query">60</span></label>
      <input type="range" id="s-query" min="5" max="240" value="60">
    </div>
    <div class="slider-row">
      <label>登簿类闭合用户数 <span class="val" id="v-register">24</span></label>
      <input type="range" id="s-register" min="2" max="120" value="24">
    </div>
    <div class="slider-row">
      <label>网上预约到达率 (tps) <span class="val" id="v-booking">6.0</span></label>
      <input type="range" id="s-booking" min="0" max="24" step="0.5" value="6">
    </div>
    <div class="slider-row">
      <label>批量上报到达率 (tps) <span class="val" id="v-bulk">1.5</span></label>
      <input type="range" id="s-bulk" min="0" max="8" step="0.1" value="1.5">
    </div>
    <div id="status"></div>
    <h2 style="margin-top:18px">各站点利用率</h2>
    <table id="util-table"><thead><tr>
      <th>站点</th><th>调度</th><th style="width:120px">利用率</th><th>队列长</th><th>驻留(s)</th>
    </tr></thead><tbody></tbody></table>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <h2>各类请求响应时间 / 吞吐</h2>
    <table id="class-table"><thead><tr>
      <th>请求类</th><th>类型</th><th>响应时间(s)</th><th>吞吐(tps)</th>
    </tr></thead><tbody></tbody></table>
  </div>
  <div class="card">
    <h2>用户数扫描: 响应时间随查询用户数变化</h2>
    <svg id="sweep" viewBox="0 0 460 240" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const SCHED_CN = {fcfs:['FCFS 先到先服务','fcfs'], ps:['PS 处理器共享','ps'], is:['IS 纯延迟','is']};
let NET = null, RESULT = null;

async function getJSON(url){ const r = await fetch(url); if(!r.ok) throw new Error((await r.json()).error); return r.json(); }

function utilColor(u){
  if(u >= 0.85) return '#ff8a5b';
  if(u >= 0.6) return '#ffd166';
  if(u >= 0.3) return '#5b9dff';
  return '#46d39a';
}

function drawNetwork(){
  const svg = $('net');
  // 按 layer 做横向分层布局
  const layers = {};
  NET.stations.forEach(s => (layers[s.layer] ??= []).push(s));
  const layerKeys = Object.keys(layers).sort();
  const pos = {};
  const xStep = 720 / (layerKeys.length + 1);
  layerKeys.forEach((lk, li) => {
    const arr = layers[lk];
    const x = 60 + (li + 1) * xStep - 40;
    arr.forEach((s, si) => {
      const y = 60 + si * 110;
      pos[s.name] = {x, y};
    });
  });
  let html = '';
  // 边
  NET.edges.forEach(e => {
    const a = pos[e.from], b = pos[e.to];
    if(!a || !b) return;
    const x1 = a.x + 130, y1 = a.y + 26, x2 = b.x, y2 = b.y + 26;
    const mx = (x1+x2)/2;
    html += `<path class="edge" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2-4},${y2}"/>`;
  });
  // 节点
  const bn = RESULT ? RESULT.bottleneck_order[0] : null;
  NET.stations.forEach(s => {
    const p = pos[s.name];
    const u = RESULT ? RESULT.stations[s.name].utilization : 0;
    const hot = (s.name === bn) ? ' bottleneck' : '';
    const [cn, cls] = SCHED_CN[s.sched];
    const pool = s.pool ? `池${s.pool}` : `${s.servers ?? '∞'}台`;
    html += `<g class="node${hot}" transform="translate(${p.x},${p.y})">
      <rect width="130" height="62" />
      <text x="10" y="20">${s.name}</text>
      <text class="util" x="10" y="42" fill="${utilColor(u)}">${(u*100).toFixed(1)}%</text>
      <text class="meta" x="64" y="42">${pool}</text>
      <text class="meta" x="10" y="56"><tspan class="badge ${cls}">${cn}</tspan></text>
    </g>`;
  });
  svg.innerHTML = html;
  // 瓶颈转移链
  if(RESULT){
    $('chain').innerHTML = '瓶颈优先级(解除后自动转移): ' +
      RESULT.bottleneck_order.map((b,i)=>
        (i===0?'<b>'+b+' (当前瓶颈)</b>':b)).join(' → ');
  }
}

function drawTables(){
  const tb = $('util-table').querySelector('tbody');
  tb.innerHTML = NET.stations.map(s => {
    const m = RESULT.stations[s.name];
    const [cn, cls] = SCHED_CN[s.sched];
    const hot = m.utilization >= 0.6 ? 'hot' : '';
    return `<tr>
      <td>${s.name}</td>
      <td><span class="badge ${cls}">${cn}</span></td>
      <td><div class="bar ${hot}"><div style="width:${Math.min(m.utilization*100,100)}%;background:${utilColor(m.utilization)}"></div></div>
          <div style="font-size:10px;color:var(--muted)">${(m.utilization*100).toFixed(1)}%</div></td>
      <td>${m.queue_length.toFixed(2)}</td>
      <td>${m.residence.toFixed(4)}</td></tr>`;
  }).join('');
  const cb = $('class-table').querySelector('tbody');
  cb.innerHTML = NET.classes.map(c => {
    const m = RESULT.classes[c.name];
    const kind = c.kind === 'closed' ? `闭合·${c.population}用户` : '开放·外部到达';
    return `<tr><td style="color:${c.color}">● ${c.name}</td><td>${kind}</td>
      <td>${m.response_time.toFixed(4)}</td><td>${m.throughput.toFixed(3)}</td></tr>`;
  }).join('');
  const st = $('status');
  if(RESULT.stable){
    st.innerHTML = '<div class="okbox">✓ 系统稳定, 所有站点利用率低于容量上限</div>';
  } else {
    st.innerHTML = '<div class="warnbox">⚠ ' + (RESULT.warnings.join('；') || '存在饱和站点') + '</div>';
  }
}

async function refresh(){
  const q = $('s-query').value, r = $('s-register').value;
  const b = $('s-booking').value, k = $('s-bulk').value;
  $('v-query').textContent = q; $('v-register').textContent = r;
  $('v-booking').textContent = (+b).toFixed(1); $('v-bulk').textContent = (+k).toFixed(1);
  RESULT = await getJSON(`/api/solve?query=${q}&register=${r}&booking=${b}&bulk=${k}`);
  drawNetwork(); drawTables();
}

async function drawSweep(){
  const data = await getJSON('/api/sweep?var=query&lo=10&hi=220&steps=22');
  const svg = $('sweep'); const W=460,H=240, pad=42;
  const xs = data.points.map(p=>p.x);
  const classes = Object.keys(data.points[0].response);
  const colors = {query:'#4C78A8',register:'#F58518',booking:'#54A24B',bulk:'#E45756'};
  let maxY = 0;
  data.points.forEach(p=>classes.forEach(c=>maxY=Math.max(maxY,p.response[c])));
  const X = v => pad + (v-Math.min(...xs))/(Math.max(...xs)-Math.min(...xs))*(W-pad-12);
  const Y = v => H-pad - (v/maxY)*(H-pad-20);
  let html = `<line x1="${pad}" y1="${H-pad}" x2="${W-12}" y2="${H-pad}" stroke="#2a3550"/>
    <line x1="${pad}" y1="12" x2="${pad}" y2="${H-pad}" stroke="#2a3550"/>
    <text x="${pad}" y="14" fill="#8595b0" font-size="10">响应(s)</text>
    <text x="${W-60}" y="${H-12}" fill="#8595b0" font-size="10">查询用户数</text>`;
  classes.forEach(c => {
    const d = data.points.map((p,i)=>(i?'L':'M')+X(p.x).toFixed(1)+','+Y(p.response[c]).toFixed(1)).join(' ');
    html += `<path d="${d}" stroke="${colors[c]||'#999'}" stroke-width="2" fill="none"/>`;
    const last = data.points[data.points.length-1];
    html += `<text x="${X(last.x)+4}" y="${Y(last.response[c])+3}" fill="${colors[c]||'#999'}" font-size="10">${c}</text>`;
  });
  // x 轴刻度
  [10,60,120,220].forEach(t=>{ if(t>=Math.min(...xs)&&t<=Math.max(...xs))
    html += `<text x="${X(t)-8}" y="${H-pad+16}" fill="#8595b0" font-size="9">${t}</text>`; });
  svg.innerHTML = html;
}

async function init(){
  NET = await getJSON('/api/network');
  ['s-query','s-register','s-booking','s-bulk'].forEach(id=>
    $(id).addEventListener('input', refresh));
  await refresh();
  drawSweep();
}
init().catch(e=>{ document.body.innerHTML = '<pre style="color:#ff8a5b;padding:24px">加载失败: '+e.message+'</pre>'; });
</script>
</body>
</html>
"""
