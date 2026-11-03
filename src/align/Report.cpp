#include "align/Report.h"

#include <fstream>
#include <stdexcept>

namespace spirula::align {
void write_report(const std::filesystem::path& output,const std::string& manifest) {
    std::ofstream out(output/"index.html");
    out << R"HTML(<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photograph alignment review</title><style>
:root{color-scheme:dark;font:15px system-ui;background:#11151b;color:#e1e8f0}
*{box-sizing:border-box}
body{max-width:1800px;margin:auto;padding:24px}
h1{font-size:28px}h2{font-size:17px;overflow-wrap:anywhere}
p{max-width:1100px;line-height:1.6;color:#aebccc}a{color:#8dc8ff}
header{border-bottom:1px solid #344151;padding-bottom:16px}
article{margin:24px 0 40px;padding:20px;background:#1b232e;border:1px solid #344151;border-radius:12px}
.controls{display:flex;align-items:center;gap:12px 18px;flex-wrap:wrap;margin:14px 0}
.controls label{display:flex;align-items:center;gap:8px;font-size:14px}
select,button{font:inherit;background:#26354a;color:#e1e8f0;border:1px solid #415570;border-radius:6px;padding:7px 10px}
select{max-width:100%}button{cursor:pointer}button:hover,button.active{background:#395778;border-color:#8fcaff}
input[type=range]{width:190px;accent-color:#76baff;cursor:ew-resize}
input[type=checkbox]{accent-color:#76baff}input:disabled{opacity:.4;cursor:default}
output{min-width:3em;font-variant-numeric:tabular-nums}
.toolbar{position:sticky;top:0;z-index:10;background:#11151bf2;backdrop-filter:blur(10px);padding:12px 14px;border:1px solid #415570;border-radius:8px;box-shadow:0 5px 20px #0005}
.endpoints{justify-content:center;margin:8px 0}.endpoints button{font-size:13px}
.comparison-figure{margin:0}
figcaption{margin:10px 0;font-size:13px;color:#bac8d8;min-height:1.5em}
.pane{overflow:auto;max-height:75vh;position:relative;background:repeating-conic-gradient(#283341 0% 25%,#18202b 0% 50%) 0/18px 18px;border-radius:5px}
.stage{position:relative;margin:0 auto;background:#000;isolation:isolate;min-height:1px}
.layer{position:absolute;inset:0;background:#000}
.layer img,.exclusions{display:block;width:100%;height:100%;object-fit:fill;user-select:none}
.exclusions{position:absolute;inset:0;opacity:0;mix-blend-mode:screen;pointer-events:none}
.divider{position:absolute;inset-block:0;width:2px;background:white;box-shadow:0 0 3px #000;pointer-events:none}
.divider span{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);display:grid;place-items:center;width:34px;height:34px;border-radius:50%;background:#111b;border:2px solid white;font-size:20px}
.wipe{touch-action:none;cursor:ew-resize}
[hidden]{display:none!important}
.score-table{width:100%;border-collapse:collapse;margin:15px 0;font-variant-numeric:tabular-nums}
.score-table td,.score-table th{text-align:left;border-bottom:1px solid #344151;padding:10px}
.metrics{font-size:13px;font-variant-numeric:tabular-nums;color:#aebccc;line-height:1.7}
.asset-links{font-size:13px}.error{color:#ffb0a6}details{margin-top:14px}summary{cursor:pointer}
footer{margin:35px 0;color:#aebccc;font-size:13px}
@media(max-width:800px){body{padding:10px}article{padding:12px}.toolbar{position:static}.controls{gap:10px}.controls label{flex-wrap:wrap}input[type=range]{width:160px}.score-table{font-size:12px}.score-table td,.score-table th{padding:5px}}
</style><header><h1>Photograph alignment review</h1><p>Choose a pair of images and compare them in one pane: blend, drag a wipe divider, or show their RGB difference. Set all views together above the photographs, or adjust each view separately. Red marks missing warp coverage; blue marks exclusions transported from the source mask.</p><p id="summary"></p><p>Alignment scores compare reserved patches with the reference splat. Native detail uses the saved original-resolution photographs; renders and diagnostic maps keep their saved working resolution. Rendered colors are never inserted into corrected photographs.</p><a href="manifest.json">Manifest and settings</a></header><main id="views"></main><footer>White training mask = usable evidence; black = ignored. Use Spirula’s “Ignore distractors” mask mode. Only missing warp coverage and transported source masks exclude pixels. Matching confidence never creates holes.</footer><script>
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
if(data.refine_cameras){const p=document.createElement('p');p.textContent=data.residual_warp?'Camera refinement followed by residual image correction. The COLMAP copy in sparse/0 contains the accepted poses, output filenames and transported 2D observations. World coordinates and georeferencing are preserved.':'Camera refinement: source photographs are unchanged. The corrected COLMAP model is in sparse/0; point coordinates and georeferencing are preserved. Compare the photograph against the initial and refined renders.';document.querySelector('header').append(p);}
if(data.refine_cameras){
 const methods=document.createElement('details');methods.innerHTML='<summary>How alignment works and how to read this review</summary><p>RoMa v2 matches the photograph to a splat render in both directions. Reliable matches are lifted into 3D using rendered ray depth to fit a rigid camera correction. Low-pass image patches provide another rigid proposal and a rotation-only proposal. The splat and world coordinate frame remain fixed.</p><p>Each proposal is rendered again. Fine/coarse normalized patch correlation can validate it. If appearance differences reject every proposal, fresh bidirectional RoMa matches can validate the geometric proposal by reducing displacement at the same photo locations. Lost matches earn no improvement. For unusable starting poses, up to four nearby registered cameras provide alternative initializations, followed by refinement and fresh validation.</p><p>The residual stage fits a smooth image warp using fresh matches, image-edge bending constraints and supported depth boundaries. Thin foreground edges add distinct, two-way gradient-profile matches perpendicular to the edge; local agreement rejects ambiguous parallel parts. Missing background depth does not discard a visible foreground edge. The normal match leaves motion along the edge free. Fine source ridges propagate the correction into poorly matched extensions; reliably matched regions keep their existing smoothing. If the extra constraints fail deformation or image checks, the original RoMa warp is tried before retaining a source photo. Final pixels are sampled directly from the original photo. Its own image audit can reject the warp while retaining an accepted camera correction. Failed corrections retain the source photograph.</p><p>Start with Corrected ↔ render and use Wipe or Flip at native detail. Original ↔ initial render shows the starting camera; Original ↔ render isolates the camera correction. Residual error maps describe the warp after camera alignment. RGB difference also shows lighting and splat-detail differences; matching scores are estimates, not ground truth.</p>';
 document.querySelector('header').append(methods);
 const counts=document.createElement('p');counts.textContent=`Exported: ${data.views.filter(v=>v.status==='ok'||v.status==='fallback').length} · cameras refined: ${data.views.filter(v=>v.camera_refined).length} · original photos retained: ${data.views.filter(v=>v.status==='fallback').length}`;document.querySelector('header').append(counts);
}
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
const toolbar=document.createElement('div');toolbar.className='controls toolbar';
toolbar.innerHTML='<strong>Set all views</strong><label>Compare <select aria-label="All views comparison"><option value="original-corrected">Original ↔ corrected</option><option value="original-render">Original ↔ render</option><option value="corrected-render">Corrected ↔ render</option><option value="corrected-mask">Corrected ↔ training mask</option><option value="corrected-coverage">Corrected ↔ warp coverage</option></select></label><label>Mode <select aria-label="All views mode"><option value="blend">Blend</option><option value="wipe">Wipe</option><option value="difference">Difference</option></select></label><label>Blend / position <input aria-label="All views blend" type="range" min="0" max="100" value="50"><output>50%</output></label>';
const root=document.querySelector('#views');root.before(toolbar);
const allPair=toolbar.querySelector('[aria-label="All views comparison"]'),allMode=toolbar.querySelector('[aria-label="All views mode"]'),allBlend=toolbar.querySelector('input');
if(data.refine_cameras)allPair.value=data.residual_warp?'corrected-render':'original-render';
const applyAll=document.createElement('button');applyAll.type='button';applyAll.textContent='Apply to all';toolbar.append(applyAll);
applyAll.addEventListener('click',()=>cards.forEach(c=>{c.setPair(allPair.value);c.setMode(allMode.value);c.setBlend(allBlend.value)}));
allPair.addEventListener('change',()=>cards.forEach(c=>c.setPair(allPair.value)));
allMode.addEventListener('change',()=>{allBlend.disabled=allMode.value==='difference';cards.forEach(c=>c.setMode(allMode.value))});
allBlend.addEventListener('input',()=>{toolbar.querySelector('output').value=allBlend.value+'%';cards.forEach(c=>c.setBlend(allBlend.value))});

for(const [i,v] of data.views.entries()){
 const card=document.createElement('article');root.append(card);
 const title=document.createElement('h2');title.textContent=`${i+1}. ${v.source}`;card.append(title);
 const available=versions(v).filter(c=>c.preview_available);
 if(!available.length){const error=document.createElement('p');error.className='error';error.textContent=v.error||'No completed correction';card.append(error);continue}
 const metrics=document.createElement('div');metrics.className='metrics';card.append(metrics);
 const controls=document.createElement('div');controls.className='controls';
 controls.innerHTML='<label>Compare <select aria-label="Image pair"></select></label><label>Mode <select aria-label="Comparison mode"><option value="blend">Blend</option><option value="wipe">Wipe</option><option value="difference">Difference</option></select></label><label><span class="range-label">Blend</span> <input aria-label="Comparison blend" type="range" min="0" max="100" value="50"><output>50%</output></label><label>Zoom <select aria-label="Comparison zoom"><option value="100">Fit</option><option value="200">200%</option><option value="400">400%</option><option value="800">800%</option><option value="pixels">Actual pixels</option></select></label><label>Photographs <select aria-label="Photograph resolution"><option value="preview">Working preview</option><option value="native">Native detail</option></select></label><label><input aria-label="Show corrected exclusions" type="checkbox"> Show corrected exclusions</label>';
 card.append(controls);
 const endpoints=document.createElement('div');endpoints.className='controls endpoints';
 const buttonA=document.createElement('button'),blink=document.createElement('button'),buttonB=document.createElement('button');
 for(const b of [buttonA,blink,buttonB])b.type='button';blink.textContent='Blink A / B';blink.title='Switch directly between the two images';endpoints.append(buttonA,blink,buttonB);card.append(endpoints);
 const figure=document.createElement('figure');figure.className='comparison-figure';
 const caption=document.createElement('figcaption');figure.append(caption);
 const pane=document.createElement('div');pane.className='pane';
 const stage=document.createElement('div');stage.className='stage';stage.style.aspectRatio=v.width+'/'+v.height;
 const layers=['a','b'].map(name=>{const layer=document.createElement('div');layer.className='layer layer-'+name;const img=document.createElement('img');img.loading='lazy';img.draggable=false;layer.append(img);stage.append(layer);return {layer,img}});
 const exclusions=document.createElement('img');exclusions.className='exclusions';exclusions.alt='Corrected-image exclusions: red is missing warp coverage; blue is the transported source mask';exclusions.loading='lazy';exclusions.draggable=false;stage.append(exclusions);
 const divider=document.createElement('div');divider.className='divider';divider.hidden=true;divider.innerHTML='<span>↔</span>';stage.append(divider);
 pane.append(stage);figure.append(pane);card.append(figure);
 const pairSelect=controls.querySelector('[aria-label="Image pair"]'),mode=controls.querySelector('[aria-label="Comparison mode"]'),range=controls.querySelector('input[type=range]'),zoom=controls.querySelector('[aria-label="Comparison zoom"]'),native=controls.querySelector('[aria-label="Photograph resolution"]'),mask=controls.querySelector('input[type=checkbox]');
 let pairs=[],pair=null,generation=0;
 const original={kind:'photo',label:'Original photo',preview:v.preview+'/original.png',native:v.original_native||(data.refine_cameras&&!data.residual_warp?v.image:null)};
 const render={kind:'render',label:data.refine_cameras?'Render at output pose':'Splat render',preview:v.preview+'/render.png'};
 const corrected=c=>({kind:'photo',label:data.refine_cameras&&!data.residual_warp?'Unchanged source photograph':data.comparison?sizeLabel(c.resolution)+' correction':'Corrected photo',preview:c.preview+'/corrected.png',native:c.image,version:c});
 function center(){
  const a=stage.getBoundingClientRect(),b=pane.getBoundingClientRect();
  return {x:a.width?(b.left+pane.clientWidth/2-a.left)/a.width:.5,y:a.height?(b.top+pane.clientHeight/2-a.top)/a.height:.5};
 }
 function layout(saved=center()){
  const n=pair?.a.kind==='photo'?0:pair?.b.kind==='photo'?1:0,img=layers[n].img;
  const aspect=img.naturalWidth&&img.naturalHeight?img.naturalWidth/img.naturalHeight:v.width/v.height;
  const fit=Math.max(1,Math.min(pane.clientWidth,window.innerHeight*.72*aspect));
  stage.style.aspectRatio=String(aspect);
  stage.style.width=(zoom.value==='pixels'?(img.naturalWidth||(native.value==='native'?v.width:fit)):fit*Number(zoom.value)/100)+'px';
  const a=stage.getBoundingClientRect(),b=pane.getBoundingClientRect();
  pane.scrollLeft+=a.left+saved.x*a.width-(b.left+pane.clientWidth/2);
  pane.scrollTop+=a.top+saved.y*a.height-(b.top+pane.clientHeight/2);
 }
 function display(){
  if(!pair)return;
  const p=Number(range.value);range.disabled=mode.value==='difference';
  layers[1].layer.style.opacity=mode.value==='blend'?p/100:1;
  layers[1].layer.style.mixBlendMode=mode.value==='difference'?'difference':'normal';
  layers[1].layer.style.clipPath=mode.value==='wipe'?`inset(0 ${100-p}% 0 0)`:'none';
  divider.hidden=mode.value!=='wipe';divider.style.left=p+'%';stage.classList.toggle('wipe',mode.value==='wipe');
  controls.querySelector('.range-label').textContent=mode.value==='wipe'?'Position':'Blend';
  controls.querySelector('output').value=mode.value==='difference'?'A − B':p+'%';
  const c=pair.maskVersion;mask.disabled=!c;exclusions.style.opacity=mask.checked&&c?'.65':'0';
  buttonA.textContent='A: '+pair.a.label;buttonB.textContent='B: '+pair.b.label;
  buttonA.classList.toggle('active',mode.value!=='difference'&&p===0);buttonB.classList.toggle('active',mode.value!=='difference'&&p===100);
  caption.textContent=mode.value==='difference'?`RGB difference · ${pair.a.label} / ${pair.b.label} · black = agreement; brightness and color differences also appear`:mode.value==='wipe'?`${pair.b.label} on the left · ${pair.a.label} on the right · drag the divider`:`${pair.a.label} ↔ ${pair.b.label} · ${p}% B`;
  if(native.value==='native'&&[pair.a,pair.b].some(x=>x.kind==='render'||x.kind==='coverage'))caption.textContent+=' · render / coverage retain their saved working resolution';
 }
 function images(){
  if(!pair)return;
  const token=++generation,saved=center();
  for(const [n,item] of [pair.a,pair.b].entries()){
   const img=layers[n].img,target=pathUrl(native.value==='native'&&item.native?item.native:item.preview);
   img.alt=item.label;
   img.onload=()=>{if(token===generation)layout()};
   img.onerror=()=>{if(token===generation){caption.textContent='Cannot load '+item.label;caption.classList.add('error')}};
   if(img.getAttribute('src')!==target)img.src=target;
  }
  const c=pair.maskVersion||available[0],target=pathUrl(c.mask_preview||c.preview+'/ignored.png');
  if(exclusions.getAttribute('src')!==target)exclusions.src=target;
  caption.classList.remove('error');layout(saved);display();
 }
 function endpoint(value){if(mode.value==='difference')mode.value='blend';range.value=value;display()}
 buttonA.addEventListener('click',()=>endpoint(0));buttonB.addEventListener('click',()=>endpoint(100));blink.addEventListener('click',()=>endpoint(Number(range.value)<50?100:0));
 pairSelect.addEventListener('change',()=>{pair=pairs.find(p=>p.id===pairSelect.value);images()});
 mode.addEventListener('change',display);range.addEventListener('input',display);mask.addEventListener('change',display);
 zoom.addEventListener('change',()=>layout());native.addEventListener('change',images);window.addEventListener('resize',()=>layout());
 const drag=e=>{const r=stage.getBoundingClientRect();range.value=Math.round(Math.max(0,Math.min(100,(e.clientX-r.left)/r.width*100)));display()};
 stage.addEventListener('pointerdown',e=>{if(mode.value!=='wipe'||e.button!==0)return;e.preventDefault();stage.setPointerCapture(e.pointerId);drag(e)});
 stage.addEventListener('pointermove',e=>{if(mode.value==='wipe'&&stage.hasPointerCapture(e.pointerId))drag(e)});
 stage.addEventListener('pointerup',e=>{if(stage.hasPointerCapture(e.pointerId))stage.releasePointerCapture(e.pointerId)});
 const links=document.createElement('p');links.className='asset-links';
 for(const version of available){const q=version.diagnostic_preview||version.preview;for(const [label,url] of [[`${sizeLabel(version.resolution)}: native image`,version.image],[`${sizeLabel(version.resolution)}: mask`,version.mask],[`${sizeLabel(version.resolution)}: warp`,version.warp_available===false?null:q+'/warp.bin']]){if(!url)continue;const a=document.createElement('a');a.textContent=label;a.href=pathUrl(url);links.append(a,document.createTextNode(' · '))}}card.append(links);
 const details=document.createElement('details'),heading=document.createElement('summary');heading.textContent='Inspect matching and warp diagnostics';details.append(heading);
 for(const version of available){
  const q=version.diagnostic_preview||version.preview,d=version.matching;
  if(d){const stats=document.createElement('p');stats.textContent=`${sizeLabel(version.resolution)} · sampled: ${d.sampled} → finite: ${d.finite} → overlap: ${d.overlap} → round trip: ${d.cycle} → displacement: ${d.bounded}. Legacy filters: opacity ${d.old_opacity} → depth ${d.old_depth} → correlation ${d.old_correlation}. Median movement ${d.median_displacement.toFixed(3)} working px; round trip ${d.median_cycle.toFixed(3)} working px. These diagnostics do not determine image validity.`;details.append(stats);}
  const maps=[['RoMa overlap: white = high','overlap'],['Round trip: green = consistent, red = 5 px or more','cycle'],['Raw render-to-photo movement','raw-flow'],['Final render-to-photo movement','deformation'],['Actual source coverage: white = covered','coverage']];
  if(version.alignment_error_maps)for(const state of ['before','after'])maps.push([`${data.residual_warp?'Residual correspondence':'Correspondence'} error ${state}: green = 0, red = 10 working px or more; gray = unverified; magenta = uncovered`,'alignment-error-'+state]);
  for(const [label,file] of maps){const caption=document.createElement('p');caption.textContent=`${sizeLabel(version.resolution)} · ${label}`;const img=document.createElement('img');img.loading='lazy';img.style.cssText='max-width:100%;display:block';img.src=pathUrl(q+'/'+file+'.png');img.alt=label;details.append(caption,img);}
 }card.append(details);
 function refresh(){
  const selected=available.filter(c=>chosen.has(c.resolution)),previous=pairSelect.value;pairs=[];
  const add=(family,id,label,a,b,maskVersion,group='Photographs and render')=>pairs.push({family,id,label,a,b,maskVersion,group});
  const suffix=c=>data.comparison?' · '+sizeLabel(c.resolution):'';
  if(!data.refine_cameras||data.residual_warp)for(const c of selected)add('original-corrected','original-corrected-'+c.resolution,'Original ↔ corrected'+suffix(c),original,corrected(c),c);
  add('original-render','original-render','Original ↔ render',original,render,null);
  if(v.camera_refined)add('initial-render','initial-render','Original ↔ initial render',original,{kind:'render',label:'Render at initial pose',preview:v.preview+'/initial-render.png'},null);
  for(const c of selected){
   add('corrected-render','corrected-render-'+c.resolution,'Corrected ↔ render'+suffix(c),corrected(c),render,c);
   add('corrected-mask','corrected-mask-'+c.resolution,'Corrected ↔ training mask'+suffix(c),corrected(c),{kind:'mask',label:'Training mask',preview:c.mask},c,'Validity');
   add('corrected-coverage','corrected-coverage-'+c.resolution,'Corrected ↔ warp coverage'+suffix(c),corrected(c),{kind:'coverage',label:'Warp coverage',preview:(c.diagnostic_preview||c.preview)+'/coverage.png'},c,'Validity');
  }
  for(let a=0;a<selected.length;a++)for(let b=a+1;b<selected.length;b++)add('corrections','corrections-'+selected[a].resolution+'-'+selected[b].resolution,`${sizeLabel(selected[a].resolution)} correction ↔ ${sizeLabel(selected[b].resolution)} correction`,corrected(selected[a]),corrected(selected[b]),selected[b],'Correction resolutions');
  pairSelect.replaceChildren();const groups=new Map();
  for(const p of pairs){if(!groups.has(p.group)){const group=document.createElement('optgroup');group.label=p.group;groups.set(p.group,group);pairSelect.append(group)}const option=document.createElement('option');option.value=p.id;option.textContent=p.label;groups.get(p.group).append(option)}
  pair=pairs.find(p=>p.id===previous)||pairs.find(p=>p.family===allPair.value)||pairs[0];pairSelect.value=pair.id;
  metrics.replaceChildren();
  for(const c of selected){const row=document.createElement('div');const scores=data.refine_cameras?`camera render NCC ${number(c.camera_audit_before)} → ${number(c.camera_audit_after)} · held-out weighted pose error ${number(c.pose_error_before,2)} → ${number(c.pose_error_after,2)} px${data.residual_warp?` · residual NCC ${number(c.audit_before)} → ${number(c.audit_after)}`:''}`:c.common_audits?c.common_audits.map(a=>`${a.resolution} common NCC ${number(a.before)} → ${number(a.after)} (${a.count} identical patches; ${a.invalid} unavailable)`).join(' · '):`reserved NCC ${number(c.audit_before)} → ${number(c.audit_after)} (${c.audit_count} scored; ${c.audit_invalid} unavailable)`;row.textContent=`${sizeLabel(c.resolution)} · usable ${(100*c.valid_fraction).toFixed(3)}% · source masks ${(100*c.source_mask_loss_fraction).toFixed(3)}% · ${scores} · native ${c.width} × ${c.height}`;if(data.refine_cameras){row.textContent+=` · method: ${c.camera_method||'unreported'} · validation: ${c.camera_status||'unreported'}`;if(c.camera_match_count)row.textContent+=` · fresh common-match error ${number(c.camera_match_before,2)} → ${number(c.camera_match_after,2)} px (${c.camera_match_count} locations)`;if(c.recovery_seed_image>=0)row.textContent+=` · recovery seed ${c.recovery_seed_image}, fresh error ${number(c.recovery_validation_error,2)} px`;}if(c.structure_fallback)row.textContent+=' · conservative RoMa warp after edge constraints were rejected';if(c.status==='fallback'){row.className='error';row.textContent+=(data.residual_warp?' · residual warp rejected: ':' · unchanged original: ')+c.fallback_reason}metrics.append(row)}
  images();
 }
 const controller={refresh,setPair:family=>{pair=pairs.find(p=>p.family===family)||pairs[0];pairSelect.value=pair.id;images()},setMode:value=>{mode.value=value;display()},setBlend:value=>{range.value=value;display()}};
 cards.push(controller);refresh();
}
function updateVersions(){cards.forEach(c=>c.refresh())}
</script></html>)HTML";
    if (!out) throw std::runtime_error("cannot write alignment report");
}
}
