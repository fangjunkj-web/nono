import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

MODEL_ID='openai/clip-vit-base-patch32'

class Matcher:
    def __init__(self, cache_dir):
        from transformers import CLIPModel, CLIPProcessor
        self.torch=__import__('torch')
        self.device='mps' if self.torch.backends.mps.is_available() else 'cpu'
        self.model=CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self.proc=CLIPProcessor.from_pretrained(MODEL_ID)
        self.cache=Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)

    def _key(self,p):
        s=p.stat(); return hashlib.sha1(f'{p.resolve()}:{s.st_mtime_ns}:{s.st_size}'.encode()).hexdigest()

    def image_embeddings(self, paths, progress=None):
        out=[]
        for i,p in enumerate(paths):
            cf=self.cache/(self._key(p)+'.npy')
            if cf.exists(): v=np.load(cf)
            else:
                im=Image.open(p).convert('RGB')
                inp=self.proc(images=im, return_tensors='pt').to(self.device)
                with self.torch.no_grad(): v=self.model.get_image_features(**inp)[0].float().cpu().numpy()
                v=v/(np.linalg.norm(v)+1e-12); np.save(cf,v)
            out.append(v)
            if progress: progress(i+1,len(paths))
        return np.stack(out)

    def text_embeddings(self, texts):
        inp=self.proc(text=texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with self.torch.no_grad(): v=self.model.get_text_features(**inp).float().cpu().numpy()
        return v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-12)

    def match(self,titles,paths,progress=None,topk=5):
        iv=self.image_embeddings(paths,progress)
        tv=self.text_embeddings(titles)
        scores=tv@iv.T
        result=[]; used={}
        for ti,row in enumerate(scores):
            order=np.argsort(-row)
            ranked=sorted(order[:min(len(order),max(30,topk*6))], key=lambda j: -(float(row[j])-0.055*used.get(int(j),0)))
            picks=ranked[:topk]
            if picks: used[int(picks[0])]=used.get(int(picks[0]),0)+1
            result.append([(paths[int(j)],float(row[j])) for j in picks])
        return result
