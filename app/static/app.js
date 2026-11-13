const labels = {Occupation:'Occupation',self_employed:'Self-employed',family_history:'Family history',Days_Indoors:'Days spent indoors',Growing_Stress:'Growing stress',Changes_Habits:'Changes in habits',Mental_Health_History:'Mental health history',Mood_Swings:'Mood swings',Coping_Struggles:'Coping struggles',Work_Interest:'Work interest',Social_Weakness:'Social weakness'};
const $ = id => document.getElementById(id);
const form = $('prediction-form'), fields = $('fields'), errorBox = $('error'), resultBox = $('result');
let lastPayload = null, reportUrl = null;
function showError(message){errorBox.textContent=message;errorBox.classList.remove('hidden');}
async function request(url, options){
  const response = await fetch(url, options);
  if(!response.ok){
    const error = await response.json().catch(()=>({}));
    const detail = Array.isArray(error.detail) ? error.detail.map(e=>`${e.loc.at(-1)}: ${e.msg}`).join('; ') : error.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response;
}
async function loadForm(){
  $('submit').disabled=true;
  const options=await (await request('/api/options')).json();
  fields.replaceChildren();
  for(const [key,values] of Object.entries(options)){
    const wrap=document.createElement('div');wrap.className='field';
    const label=document.createElement('label');label.htmlFor=key;label.textContent=labels[key]||key;
    const select=document.createElement('select');select.id=key;select.name=key;select.required=true;
    const placeholder=new Option('Select '+label.textContent,'',true,true);placeholder.disabled=true;select.add(placeholder);
    values.forEach(v=>select.add(new Option(v,v)));wrap.append(label,select);fields.append(wrap);
  }
  $('submit').disabled=false;
}
async function loadStats(){
  const stats=await (await request('/api/stats')).json();
  for(const [id,key] of [['rows','rows'],['cols','columns'],['missing','missing_values'],['duplicates','duplicate_rows']]) $(id).textContent=Number(stats[key]).toLocaleString();
  const m=stats.metrics;
  const entries=[['Accuracy',m.accuracy],['Macro F1',m.f1_macro],['ROC AUC',m.roc_auc],['Weighted F1',m.f1_weighted]];
  $('metrics').replaceChildren();
  for(const [name,value] of entries){const box=document.createElement('div');box.className='metric';const title=document.createElement('span'), score=document.createElement('b');title.textContent=name;score.textContent=(value*100).toFixed(1)+'%';box.append(title,score);$('metrics').append(box);}
  $('evaluation-details').textContent=`Selected model: ${m.selected_model}. ${m.training_rows.toLocaleString()} training rows; ${m.test_rows.toLocaleString()} held-out rows. ${m.split_method}. Shared timestamp groups: ${m.group_overlap}. ${m.limitations}`;
  const comparison=$('model-comparison');comparison.replaceChildren();
  for(const [name,scores] of Object.entries(m.cross_validation)){const p=document.createElement('p');p.textContent=`${name}: CV macro F1 ${(scores.f1_macro*100).toFixed(1)}%, balanced accuracy ${(scores.balanced_accuracy*100).toFixed(1)}%`;comparison.append(p);}
  const heading=document.createElement('h3');heading.textContent='Held-out results by class';comparison.append(heading);
  for(const name of m.classes){const r=m.classification_report[name], p=document.createElement('p');p.textContent=`${name}: precision ${(r.precision*100).toFixed(1)}%, recall ${(r.recall*100).toFixed(1)}%, F1 ${(r['f1-score']*100).toFixed(1)}%, support ${r.support.toLocaleString()}`;comparison.append(p);}
  const matrix=document.createElement('p');matrix.textContent=`Confusion matrix (rows: actual; columns: predicted; order: ${m.classes.join(', ')}): ${m.confusion_matrix.map(row=>row.join(' / ')).join('; ')}`;comparison.append(matrix);
}
async function loadPreview(){
  const preview=await (await request('/api/data-preview?limit=8')).json();
  $('table-head').replaceChildren();$('table-body').replaceChildren();
  const head=document.createElement('tr');
  preview.columns.forEach(c=>{const cell=document.createElement('th');cell.textContent=c;head.append(cell);});$('table-head').append(head);
  preview.rows.forEach(r=>{const row=document.createElement('tr');preview.columns.forEach(c=>{const cell=document.createElement('td');cell.textContent=String(r[c]??'');row.append(cell);});$('table-body').append(row);});
}
form.addEventListener('submit',async event=>{
  event.preventDefault();errorBox.classList.add('hidden');resultBox.classList.add('hidden');
  const button=$('submit');button.disabled=true;button.textContent='Generating...';lastPayload=null;
  try{
    const payload=Object.fromEntries(new FormData(form));
    const data=await (await request('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();
    lastPayload=payload;
    $('result-label').textContent=data.prediction;
    $('confidence').textContent=(data.confidence*100).toFixed(1)+'%';$('confidence-bar').style.width=(data.confidence*100)+'%';
    $('class-probs').replaceChildren();
    for(const [name,value] of Object.entries(data.probabilities)){const span=document.createElement('span');span.textContent=`${name}: ${(value*100).toFixed(1)}%`;$('class-probs').append(span);}
    resultBox.classList.remove('hidden');resultBox.scrollIntoView({behavior:'smooth',block:'center'});
  }catch(error){showError(error.message);}finally{button.disabled=false;button.textContent='Generate prediction';}
});
$('report-link').addEventListener('click',async event=>{
  event.preventDefault();if(!lastPayload)return;
  const button=$('report-link');button.disabled=true;
  try{
    const response=await request('/api/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lastPayload)});
    if(reportUrl)URL.revokeObjectURL(reportUrl);
    reportUrl=URL.createObjectURL(await response.blob());const link=document.createElement('a');link.href=reportUrl;link.download='mental_health_survey_report.pdf';link.click();
  }catch(error){showError(error.message);}finally{button.disabled=false;}
});
form.addEventListener('reset',()=>{lastPayload=null;resultBox.classList.add('hidden');errorBox.classList.add('hidden');if(reportUrl){URL.revokeObjectURL(reportUrl);reportUrl=null;}});
form.addEventListener('change',()=>{lastPayload=null;resultBox.classList.add('hidden');});
loadForm().catch(error=>{fields.textContent='Could not load form options.';showError(error.message);});
loadStats().catch(error=>{$('metrics').textContent='Statistics unavailable: '+error.message;});
loadPreview().catch(error=>{$('table-body').textContent='Preview unavailable: '+error.message;});
