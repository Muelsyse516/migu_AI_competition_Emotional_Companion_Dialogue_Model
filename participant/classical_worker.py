import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(ROOT/'vendor'))
import joblib
config=json.loads((ROOT/'config.json').read_text())
m=joblib.load(ROOT/config['classical_model'])
base=m.get('base_model',m) if isinstance(m,dict) and isinstance(m.get('base_model'),dict) else m
style_override=m.get('style_override_model') if isinstance(m,dict) else None
rows=[json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
timings=[]
with Path(sys.argv[2]).open('w',encoding='utf-8') as handle:
 for r in rows:
  started=time.perf_counter()
  user=[t['content'] for t in r['history'] if t['role']=='user']
  ex=base['emotion_vectorizer'].transform(['\n'.join(user[-3:-1]+[user[-1]]*3)])
  px=base['profile_vectorizer'].transform(['\n'.join(user)])
  override_px=style_override['profile_vectorizer'].transform(['\n'.join(user)]) if style_override else None
  emotion=str(base['emotion'].predict(ex)[0])
  profile={}
  for f,model in base['profiles'].items():
   if style_override and f=='style':continue
   values=model.predict(px)[0]
   profile[f]=[label for j,label in enumerate(base['labels'][f]) if values[j]]
  if style_override:
   values=style_override['profiles']['style'].predict(override_px)[0]
   profile['style']=[label for j,label in enumerate(style_override['labels']['style']) if values[j]]
  elapsed=(time.perf_counter()-started)*1000;timings.append(elapsed)
  handle.write(json.dumps({'id':r['id'],'emotion_label':emotion,'user_profile':profile,'cpu_inference_ms':elapsed},ensure_ascii=False)+'\n')
Path(sys.argv[3]).write_text(json.dumps({'inference_ms':sum(timings),'latencies_ms':timings,'samples':len(rows),'scope':'individual sample history processing, vectorization and prediction; model loading excluded'}))
