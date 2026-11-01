#include "align/Report.h"

#include <fstream>
#include <stdexcept>

namespace spirula::align {
void write_report(const std::filesystem::path& output,const std::string& manifest) {
    std::ofstream out(output/"index.html");
    out << R"HTML(<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photograph alignment review</title><style>
:root{color-scheme:dark;font:15px system-ui;background:#11151b;color:#e1e8f0}body{max-width:1800px;margin:auto;padding:24px}h1{font-size:28px}p{max-width:1000px;line-height:1.6;color:#aebccc}a{color:#8dc8ff}header{border-bottom:1px solid #344151;padding-bottom:20px}article{margin:28px 0 48px;padding:20px;background:#1b232e;border:1px solid #344151;border-radius:12px}h2{font-size:18px;overflow-wrap:anywhere}.controls{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin:16px 0}.controls label{display:flex;align-items:center;gap:8px}input[type=range]{width:210px;accent-color:#76baff}.pair{display:grid;grid-template-columns:repeat(var(--panes,2),minmax(0,1fr));gap:12px}.pane{overflow:auto;max-height:80vh;background:repeating-conic-gradient(#283341 0% 25%,#18202b 0% 50%) 0/18px 18px}.stage{position:relative;width:100%}.stage img{display:block;width:100%;height:auto}.stage .render,.stage .mask{position:absolute;inset:0;width:100%;height:100%}.render{opacity:.5}.mask{opacity:0;mix-blend-mode:screen}figcaption{margin:6px 0;min-height:2.5em;color:#bac8d8}.score-table{width:100%;border-collapse:collapse;margin:15px 0;font-variant-numeric:tabular-nums}.score-table td,.score-table th{text-align:left;border-bottom:1px solid #344151;padding:10px}.metrics{font-variant-numeric:tabular-nums;color:#aebccc;line-height:1.7}.error{color:#ffb0a6}button{background:#31445e;color:white;border:0;border-radius:5px;padding:7px;cursor:pointer}footer{margin:35px 0;color:#aebccc}@media(max-width:800px){.pair{grid-template-columns:1fr}body{padding:10px}}
</style><header><h1>Photograph alignment review</h1><p>Original and corrected photographs at the same camera pose. Move the blend slider from photograph to splat render, zoom into detail, or inspect native photograph detail. Both panels scroll together. Red marks absent warp coverage; blue marks exclusions from the original mask. Corrections use original photographic pixels; rendered colors are never inserted into the output.</p><p id="summary"></p><p>Correlation scores check reserved patches within these photographs against the reference model. They do not measure independent 3D accuracy. Matching and the reference render use working resolution; output photographs retain native dimensions. Native detail loads the original-resolution photographs. Excluded pixels in native RGB output appear black; the mask overlay identifies their cause.</p><a href="manifest.json">Manifest and settings</a></header><main id="views"></main><footer>White training mask = usable evidence; black = ignored. Use Spirula’s “Ignore distractors” mask mode. Only missing warp coverage and transported source masks exclude pixels. Matching confidence never creates holes.</footer><script>
const data=)HTML";
    for (char c:manifest) {
        if (c=='<') out << "\\u003c";
        else if (c=='>') out << "\\u003e";
        else if (c=='&') out << "\\u0026";
        else out << c;
    }
    out << R"HTML(;
const pathUrl=p=>p.split('/').map(encodeURIComponent).join('/');
const variants=data.variants||[{resolution:data.max_resolution}],resolutions=variants.map(v=>v.resolution);
const chosen=new Set([resolutions.includes(1280)?1280:resolutions[0]]),cards=[];
const mean=values=>{const valid=values.filter(Number.isFinite);return valid.length?valid.reduce((s,v)=>s+v,0)/valid.length:null};
const sizeLabel=r=>r===0?'Native':r+' px';
const number=(v,n=4)=>v==null?'—':v.toFixed(n);
const versions=v=>(v.versions||[v]).map(c=>({...c,resolution:c.resolution??data.max_resolution}));
const summary=document.querySelector('#summary');
summary.textContent=`Selected: ${data.views.length} / ${data.population} · seed ${data.seed} · reference ${data.reference}`;
if(data.comparison){const p=document.createElement('p');p.textContent='Every version is compared with the same highest-resolution reference render. Common NCC and gradient scores use identical patch locations and shared valid support at each evaluation size. Scores from the individual runs are available in their manifests.';document.querySelector('header').append(p);}
const global=document.createElement('div');global.className='controls';global.id='resolution-controls';
if(data.comparison){
 const label=document.createElement('strong');label.textContent='Show corrections:';global.append(label);
 for(const resolution of resolutions){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=chosen.has(resolution);input.dataset.resolution=resolution;input.setAttribute('aria-label',`Show ${sizeLabel(resolution)} correction`);input.addEventListener('change',()=>{input.checked?chosen.add(resolution):chosen.delete(resolution);updateVersions()});label.append(input,document.createTextNode(`${sizeLabel(resolution)}${resolution===640?' (½)':resolution===1280?' (1×)':resolution===2560?' (2×)':''}`));global.append(label);}
 const cycle=document.createElement('button');cycle.type='button';cycle.textContent='Cycle resolutions';cycle.addEventListener('click',()=>{const current=resolutions.findIndex(r=>chosen.has(r)),next=resolutions[(current+1)%resolutions.length];chosen.clear();chosen.add(next);global.querySelectorAll('input').forEach(i=>i.checked=chosen.has(+i.dataset.resolution));updateVersions()});global.append(cycle);document.querySelector('header').append(global);
}
function summaryTable(){
 if(!data.comparison)return;
 const table=document.createElement('table');table.className='score-table';table.innerHTML='<thead><tr><th>Working size</th><th>Corrected / unchanged</th><th>Usable pixels</th><th>Common NCC · 1280</th><th>Common NCC · 2560</th><th>Gradient error · 2560 ↓</th><th>Fresh run</th></tr></thead>';
 const body=document.createElement('tbody');table.append(body);
 for(const r of resolutions){const list=data.views.flatMap(v=>versions(v).filter(c=>c.resolution===r));const cells=[sizeLabel(r),`${list.filter(v=>v.status==='ok').length} / ${list.filter(v=>v.status==='fallback').length}`,number(mean(list.map(v=>100*v.valid_fraction)),3)+'%'];
 for(const size of [1280,2560]){const scores=list.map(v=>v.common_audits?.find(a=>a.resolution===size));cells.push(number(mean(scores.map(a=>a?.count?a.after:undefined))));}
 cells.push(number(mean(list.map(v=>v.common_audits?.find(a=>a.resolution===2560&&a.count)?.gradient_after))));const timing=variants.find(v=>v.resolution===r)?.timing;cells.push(timing?`${timing.timing?'≈':''}${timing.wall_seconds.toFixed(1)} s`:'— (existing baseline)');const row=document.createElement('tr');for(const value of cells){const td=document.createElement('td');td.textContent=value;row.append(td)}body.append(row);}
 document.querySelector('header').append(table);
}
summaryTable();
const root=document.querySelector('#views');
for(const [i,v] of data.views.entries()){
 const card=document.createElement('article');root.append(card);const title=document.createElement('h2');title.textContent=`${i+1}. ${v.source}`;card.append(title);
 const choices=versions(v),available=choices.filter(c=>c.preview_available);
 if(!available.length){const error=document.createElement('p');error.className='error';error.textContent=v.error||'No completed correction';card.append(error);continue}
 const metrics=document.createElement('div');metrics.className='metrics';card.append(metrics);
 const controls=document.createElement('div');controls.className='controls';controls.innerHTML='<label>Photo ↔ render <input aria-label="Photograph to render blend" type="range" min="0" max="100" value="50"><output>50%</output></label><label>Zoom <select aria-label="Comparison zoom"><option value="100">Fit</option><option value="200">200%</option><option value="400">400%</option><option value="800">800%</option><option value="pixels">Actual pixels</option></select></label><label>Photographs <select aria-label="Photograph resolution"><option value="preview">Working preview</option><option value="native">Native detail</option></select></label><label><input type="checkbox"> Show excluded pixels</label><button type="button">Blink photographs / render</button>';card.append(controls);
 const pair=document.createElement('div');pair.className='pair';card.append(pair);const panes=[];
 function pane(name,file,version){
  const figure=document.createElement('div');figure.className='comparison-figure';const cap=document.createElement('figcaption');cap.textContent=name+' + reference';figure.append(cap);
  const scroller=document.createElement('div');scroller.className='pane';const stage=document.createElement('div');stage.className='stage';
  for(const [cls,src] of [['photo',file],['render',v.preview+'/render.png'],['mask',version?(version.mask_preview||version.preview+'/ignored.png'):null]]){if(!src)continue;const img=document.createElement('img');img.className=cls;img.src=pathUrl(src);img.alt=cls==='photo'?name:cls==='render'?'Reference splat render':'Corrected-image exclusions';img.loading='lazy';stage.append(img)}
  scroller.append(stage);figure.append(scroller);pair.append(figure);const item={figure,scroller,stage,version};panes.push(item);return item;
 }
 pane('Original photograph',v.preview+'/original.png',null);
 for(const version of available)pane(`${sizeLabel(version.resolution)} correction`,version.preview+'/corrected.png',version);
 const range=controls.querySelector('input[type=range]'),zoom=controls.querySelector('[aria-label="Comparison zoom"]'),native=controls.querySelector('[aria-label="Photograph resolution"]'),mask=controls.querySelector('input[type=checkbox]');
 const blend=()=>{pair.querySelectorAll('.render').forEach(e=>e.style.opacity=range.value/100);controls.querySelector('output').value=range.value+'%'};range.addEventListener('input',blend);
 function alignScroll(){const original=panes[0].scroller;for(const p of panes)if(p.version&&!p.figure.hidden){p.scroller.scrollLeft=original.scrollLeft;p.scroller.scrollTop=original.scrollTop}}
 const applyZoom=()=>{for(const p of panes){const img=p.stage.querySelector('.photo');p.stage.style.width=zoom.value==='pixels'?Math.max(p.scroller.clientWidth,img.naturalWidth)+'px':zoom.value+'%'}};
 function images(){for(const p of panes){if(p.figure.hidden)continue;const version=p.version;const file=native.value==='native'?(version?.image||v.original_native):null;const target=file||((version?.preview||v.preview)+(version?'/corrected.png':'/original.png'));const img=p.stage.querySelector('.photo');if(img.getAttribute('src')!==pathUrl(target))img.src=pathUrl(target);img.onload=()=>{applyZoom();requestAnimationFrame(alignScroll)};}applyZoom();}
 zoom.addEventListener('change',applyZoom);native.addEventListener('change',images);mask.addEventListener('change',()=>pair.querySelectorAll('.mask').forEach(s=>s.style.opacity=mask.checked?'.65':'0'));
 controls.querySelector('button').addEventListener('click',()=>{range.value=range.value<50?100:0;blend()});
 let syncing=false;for(const p of panes)p.scroller.addEventListener('scroll',()=>{if(syncing||p.figure.hidden)return;syncing=true;for(const other of panes)if(other!==p&&!other.figure.hidden){other.scroller.scrollLeft=p.scroller.scrollLeft;other.scroller.scrollTop=p.scroller.scrollTop}requestAnimationFrame(()=>syncing=false)});
 const links=document.createElement('p');for(const version of available){const q=version.diagnostic_preview||version.preview;for(const [label,url] of [[`${sizeLabel(version.resolution)}: native image`,version.image],[`${sizeLabel(version.resolution)}: mask`,version.mask],[`${sizeLabel(version.resolution)}: warp`,q+'/warp.bin']]){if(!url)continue;const a=document.createElement('a');a.textContent=label;a.href=pathUrl(url);links.append(a,document.createTextNode(' · '))}}card.append(links);
 const details=document.createElement('details');const heading=document.createElement('summary');heading.textContent='Inspect matching and warp diagnostics';details.append(heading);
 for(const version of available){const q=version.diagnostic_preview||version.preview,d=version.matching;if(d){const stats=document.createElement('p');stats.textContent=`${sizeLabel(version.resolution)} · sampled: ${d.sampled} → finite: ${d.finite} → overlap: ${d.overlap} → round trip: ${d.cycle} → displacement: ${d.bounded}. Legacy filters: opacity ${d.old_opacity} → depth ${d.old_depth} → correlation ${d.old_correlation}. Median movement ${d.median_displacement.toFixed(3)} working px; round trip ${d.median_cycle.toFixed(3)} working px. These are matching diagnostics; they do not determine image validity.`;details.append(stats);}
 for(const [label,file] of [['RoMa overlap: white = high','overlap'],['Round trip: green = consistent, red = 5 px or more','cycle'],['Raw render-to-photo movement','raw-flow'],['Final render-to-photo movement','deformation'],['Actual source coverage: white = covered','coverage']]){const caption=document.createElement('p');caption.textContent=`${sizeLabel(version.resolution)} · ${label}`;const img=document.createElement('img');img.loading='lazy';img.style.cssText='max-width:100%;display:block';img.src=pathUrl(q+'/'+file+'.png');img.alt=label;details.append(caption,img);}}card.append(details);
 function refresh(){
  let visible=1;for(const p of panes)if(p.version){p.figure.hidden=!chosen.has(p.version.resolution);visible+=!p.figure.hidden}pair.style.setProperty('--panes',visible);
  metrics.replaceChildren();for(const c of available){if(!chosen.has(c.resolution))continue;const row=document.createElement('div');let scores=c.common_audits?c.common_audits.map(a=>`${a.resolution} common NCC ${number(a.before)} → ${number(a.after)} (${a.count} identical patches; ${a.invalid} unavailable)`).join(' · '):`reserved NCC ${number(c.audit_before)} → ${number(c.audit_after)} (${c.audit_count} scored; ${c.audit_invalid} unavailable)`;row.textContent=`${sizeLabel(c.resolution)} · usable ${(100*c.valid_fraction).toFixed(3)}% · source masks ${(100*c.source_mask_loss_fraction).toFixed(3)}% · confidence exclusions 0 · ${scores} · native ${c.width} × ${c.height}`;if(c.status==='fallback'){row.className='error';row.textContent+=' · unchanged original: '+c.fallback_reason}metrics.append(row)}images();requestAnimationFrame(alignScroll);
 }
 cards.push(refresh);refresh();
}
function updateVersions(){cards.forEach(refresh=>refresh())}
</script></html>)HTML";
    if (!out) throw std::runtime_error("cannot write alignment report");
}
}
