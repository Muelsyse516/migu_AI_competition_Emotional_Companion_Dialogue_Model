"""Profile-only encoder experiment. Calibration uses training conversations only."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json, random, time, hashlib
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed

ROOT = Path('/mnt/storage/starter/profile_bert')
OUT = Path('/mnt/storage/results/profile-bert-v1')
DATA = Path('/mnt/storage/starter/prepared')
BASE = '/mnt/storage/models/bert-base-chinese'
FIELDS = {
 'personality_traits': ['extroverted','introverted','open','conservative','high_conscientiousness','casual','agreeable','assertive','emotionally_stable','sensitive'],
 'interests': ['study_exam','programming_technology','reading_writing','film_animation','music','games','sports_fitness','travel_outdoor','pets','social','career_development','art_design'],
 'style': ['brief','detailed','colloquial','formal','direct','indirect','humorous','rational','high_emotional_expression','low_emotional_expression','emoji_user'],
}
LABELS = [(f,l) for f,ls in FIELDS.items() for l in ls]
def rows(p):
 return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def encode(tok, samples):
 ids=[]; masks=[]
 for r in samples:
  text='\n'.join(t['content'] for t in r['history'] if t['role']=='user')
  x=tok.encode(text,add_special_tokens=False)
  # Preserve both early interests and recent expression when context is long.
  if len(x)>510: x=x[:96]+x[-414:]
  x=[tok.cls_token_id]+x+[tok.sep_token_id]
  masks.append([1]*len(x)+[0]*(512-len(x)))
  ids.append(x+[tok.pad_token_id]*(512-len(x)))
 return torch.tensor(ids),torch.tensor(masks)
def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 samples=rows(DATA/'train_inference.jsonl'); gold={r['id']:r for r in rows(DATA/'train_answers.jsonl')}
 conv=sorted({r['conversation_id'] for r in samples}); random.Random(42).shuffle(conv)
 calibration=set(conv[:len(conv)//5])
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 ids,masks=encode(tok,samples)
 targets=torch.tensor([[float(l in gold[r['id']]['user_profile'][f]) for f,l in LABELS] for r in samples])
 split=torch.tensor([r['conversation_id'] in calibration for r in samples])
 assert not {r['conversation_id'] for r,c in zip(samples,split) if c} & {r['conversation_id'] for r,c in zip(samples,split) if not c}
 torch.save({'ids':ids,'masks':masks,'targets':targets,'calibration':split},ROOT/'tokens.pt')
 (ROOT/'data_report.json').write_text(json.dumps({'samples':len(samples),'train_samples':int((~split).sum()),'calibration_samples':int(split.sum()),'calibration_scope':'20% of training conversations only; no validation labels','seed':42,'max_tokens':512,'context_policy':'User utterances only; first 96 and last 414 content tokens for overlong histories','labels':LABELS},ensure_ascii=False,indent=2))
 print('PREPARED',len(samples),flush=True)
def infer(model,ids,masks):
 result=[]
 model.eval()
 with torch.inference_mode():
  for i in range(0,len(ids),32):
   with torch.autocast('cuda',dtype=torch.bfloat16):
    logits=model(input_ids=ids[i:i+32].to('cuda'),attention_mask=masks[i:i+32].to('cuda')).logits
   result.append(logits.float().sigmoid().cpu())
 return torch.cat(result)
def metrics(pred,target):
 p=pred.bool(); g=target.bool(); tp=int((p&g).sum()); fp=int((p&~g).sum()); fn=int((~p&g).sum())
 nonempty=g.any(1)
 return {'set_exact':float((p==g).all(1).float().mean()),'micro_f1':2*tp/max(1,2*tp+fp+fn),'micro_precision':tp/max(1,tp+fp),'micro_recall':tp/max(1,tp+fn),'nonempty_gold_count':int(nonempty.sum()),'nonempty_gold_exact':int(((p==g).all(1)&nonempty).sum()),'predicted_empty':int((~p.any(1)).sum())}
def calibrate(prob,target):
 result={}; cursor=0; objective=0
 for f,ls in FIELDS.items():
  q=prob[:,cursor:cursor+len(ls)]; y=target[:,cursor:cursor+len(ls)]
  options=[]
  for threshold in [.2,.3,.4,.5,.6,.7,.8]:
   m=metrics(q>=threshold,y); score=.5*m['set_exact']+.5*m['micro_f1']
   options.append((score,threshold,m))
  score,t,m=max(options,key=lambda z:(z[0],z[1])); result[f]={'threshold':t,**m}; objective+=score/3; cursor+=len(ls)
 return objective,result
def train():
 assert torch.cuda.is_available(),'GPU unavailable'
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'model.safetensors').exists(),'Do not overwrite completed model'
 set_seed(42); d=torch.load(ROOT/'tokens.pt',weights_only=True); train=~d['calibration']; cal=d['calibration']
 model=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=len(LABELS),ignore_mismatched_sizes=True,local_files_only=True,attn_implementation='sdpa').to('cuda')
 positives=d['targets'][train].sum(0); n=int(train.sum())
 weights=((n-positives)/positives.clamp_min(1)).sqrt().clamp(1,8).to('cuda')
 loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weights)
 optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
 loader=DataLoader(TensorDataset(d['ids'][train],d['masks'][train],d['targets'][train]),batch_size=16,shuffle=True)
 best=-1; started=time.time(); total=len(loader)*2; step=0
 for epoch in range(2):
  model.train()
  for ids,masks,targets in loader:
   step+=1; lr=2e-5*min(step/60,max(0,(total-step)/max(1,total-60)))
   for group in optimizer.param_groups: group['lr']=lr
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    logits=model(input_ids=ids.to('cuda'),attention_mask=masks.to('cuda')).logits
    loss=loss_fn(logits.float(),targets.to('cuda'))
   assert torch.isfinite(loss),'Nonfinite loss'
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
   if step%50==0: print(json.dumps({'step':step,'total':total,'epoch':epoch+1,'loss':loss.item(),'elapsed_s':time.time()-started}),flush=True)
  prob=infer(model,d['ids'][cal],d['masks'][cal]); objective,thresholds=calibrate(prob,d['targets'][cal])
  print('TRAINING_CALIBRATION',epoch+1,objective,json.dumps(thresholds),flush=True)
  if objective>best:
   best=objective; model.save_pretrained(OUT); AutoTokenizer.from_pretrained(BASE,local_files_only=True).save_pretrained(OUT)
   (OUT/'selection.json').write_text(json.dumps({'epoch':epoch+1,'training_calibration_objective':objective,'objective_definition':'0.5 set exact + 0.5 micro F1, averaged over fields; experimental, not official score','thresholds':thresholds,'data_report':str(ROOT/'data_report.json')},indent=2))
 print('TRAINING_FINISHED',flush=True)
def evaluate():
 samples=rows(DATA/'dev200_inference.jsonl'); tok=AutoTokenizer.from_pretrained(OUT,local_files_only=True)
 ids,masks=encode(tok,samples); model=AutoModelForSequenceClassification.from_pretrained(OUT,local_files_only=True,attn_implementation='sdpa').to('cuda')
 prob=infer(model,ids,masks); selection=json.loads((OUT/'selection.json').read_text()); predictions=[]
 for i,r in enumerate(samples):
  profile={}; cursor=0
  for f,ls in FIELDS.items():
   t=selection['thresholds'][f]['threshold']; profile[f]=[l for j,l in enumerate(ls) if prob[i,cursor+j]>=t]; cursor+=len(ls)
  predictions.append({'id':r['id'],'user_profile':profile})
 (OUT/'dev200_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
 # Answers are opened only after model selection and prediction have finished.
 gold={r['id']:r for r in rows(DATA/'dev200_answers.jsonl')}; report={}
 for f,ls in FIELDS.items():
  p=torch.tensor([[l in r['user_profile'][f] for l in ls] for r in predictions]); g=torch.tensor([[l in gold[r['id']]['user_profile'][f] for l in ls] for r in predictions]); report[f]=metrics(p,g)
 (OUT/'dev200_metrics.json').write_text(json.dumps(report,indent=2)); print('DEV_METRICS',json.dumps(report),flush=True)
if __name__=='__main__':
 import sys
 {'prepare':prepare,'train':train,'evaluate':evaluate}[sys.argv[1]]()
