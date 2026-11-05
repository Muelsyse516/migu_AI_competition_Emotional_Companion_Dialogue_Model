"""Verify release parts and reconstruct the Docker archive without extra gzip copy."""
import json,hashlib,gzip,io,os
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 m=json.loads((R/'release-manifest.json').read_text())
 names=[p['name'] for p in m['parts']]
 assert len(names)==len(set(names)) and all(Path(n).name==n for n in names)
 assert Path(m['archive']).name==m['archive']
 dst=R/m['archive'];temp=R/(m['archive']+'.partial')
 if dst.exists():
  assert dst.stat().st_size==m['archive_bytes'] and sha(dst)==m['archive_sha256'],'Existing archive differs; will not overwrite'
  print('Existing archive verified:',dst.name);return
 for p in m['parts']:
  f=R/p['name'];assert f.stat().st_size==p['bytes'] and sha(f)==p['sha256'],'Part verification failed: '+p['name']
  print('Verified:',p['name'],flush=True)
 class Parts(io.RawIOBase):
  def __init__(self):self.i=0;self.f=None
  def readable(self):return True
  def readinto(self,b):
   while True:
    if self.f is None:
     if self.i==len(names):return 0
     self.f=(R/names[self.i]).open('rb');self.i+=1
    n=self.f.readinto(b)
    if n:return n
    self.f.close();self.f=None
  def close(self):
   if self.f:self.f.close()
   super().close()
 h=hashlib.sha256();n=0
 with io.BufferedReader(Parts(),8*1024*1024) as stream,gzip.GzipFile(fileobj=stream) as gz,temp.open('xb') as out:
  for b in iter(lambda:gz.read(8*1024*1024),b''):
   out.write(b);h.update(b);n+=len(b)
 assert n==m['archive_bytes'] and h.hexdigest()==m['archive_sha256'],'Archive verification failed'
 temp.rename(dst);print('Docker archive verified:',dst.name)
if __name__=='__main__':main()
