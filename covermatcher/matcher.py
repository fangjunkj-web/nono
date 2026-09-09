import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_ID = 'openai/clip-vit-base-patch32'


class Matcher:
    def __init__(self, cache_dir):
        from transformers import CLIPModel, CLIPProcessor
        import torch

        self.torch = torch
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()
        self.proc = CLIPProcessor.from_pretrained(MODEL_ID)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.library_cache = self.cache.parent / 'libraries'
        self.library_cache.mkdir(parents=True, exist_ok=True)

    def _file_sig(self, p: Path):
        try:
            s = p.stat()
            return f'{p.resolve()}|{s.st_size}|{s.st_mtime_ns}'
        except OSError:
            return str(p.resolve())

    def _library_id(self, paths):
        if not paths:
            return 'empty'
        try:
            root = Path(os.path.commonpath([str(p.resolve()) for p in paths]))
        except Exception:
            root = paths[0].parent.resolve()
        return hashlib.sha1(str(root).encode('utf-8', 'ignore')).hexdigest()

    def _index_path(self, paths):
        return self.library_cache / f'{self._library_id(paths)}.npz'

    def _load_index(self, paths):
        idx_path = self._index_path(paths)
        if not idx_path.exists():
            return {}, idx_path
        try:
            data = np.load(idx_path, allow_pickle=False)
            old_paths = data['paths'].tolist()
            old_sigs = data['sigs'].tolist()
            old_vecs = data['vecs'].astype(np.float32, copy=False)
            mapping = {
                str(p): (str(sig), old_vecs[i])
                for i, (p, sig) in enumerate(zip(old_paths, old_sigs))
            }
            return mapping, idx_path
        except Exception:
            return {}, idx_path

    def _save_index(self, idx_path, rows):
        if not rows:
            return
        paths = np.asarray([str(p) for p, _, _ in rows], dtype='U')
        sigs = np.asarray([sig for _, sig, _ in rows], dtype='U')
        vecs = np.stack([v for _, _, v in rows]).astype(np.float32, copy=False)
        tmp = idx_path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, paths=paths, sigs=sigs, vecs=vecs)
        tmp.replace(idx_path)

    def _safe_open(self, p):
        try:
            with Image.open(p) as im:
                im.load()
                return im.convert('RGB').copy()
        except Exception:
            return None

    def _embed_new_images(self, items, progress=None, batch_size=8):
        rows = []
        done = 0
        total = len(items)

        for start in range(0, total, batch_size):
            batch_items = items[start:start + batch_size]
            good_paths, good_sigs, good_images = [], [], []

            for p, sig in batch_items:
                im = self._safe_open(p)
                if im is not None:
                    good_paths.append(p)
                    good_sigs.append(sig)
                    good_images.append(im)
                done += 1
                if progress:
                    progress(done, total)

            if not good_images:
                continue

            try:
                inp = self.proc(images=good_images, return_tensors='pt', padding=True).to(self.device)
                with self.torch.inference_mode():
                    vecs = self.model.get_image_features(**inp).float().cpu().numpy()
                vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
                for p, sig, v in zip(good_paths, good_sigs, vecs):
                    rows.append((p, sig, v.astype(np.float32, copy=False)))
            except Exception:
                # Fall back to one-by-one so one unusual image cannot crash a whole library.
                for p, sig, im in zip(good_paths, good_sigs, good_images):
                    try:
                        inp = self.proc(images=im, return_tensors='pt').to(self.device)
                        with self.torch.inference_mode():
                            v = self.model.get_image_features(**inp)[0].float().cpu().numpy()
                        v = v / (np.linalg.norm(v) + 1e-12)
                        rows.append((p, sig, v.astype(np.float32, copy=False)))
                    except Exception:
                        continue

            del good_images
            if self.device == 'mps':
                try:
                    self.torch.mps.empty_cache()
                except Exception:
                    pass

        return rows

    def image_embeddings(self, paths, progress=None):
        old, idx_path = self._load_index(paths)
        current = []
        missing = []

        for p in paths:
            sig = self._file_sig(p)
            hit = old.get(str(p))
            if hit and hit[0] == sig:
                current.append((p, sig, hit[1]))
            else:
                missing.append((p, sig))

        if missing:
            new_rows = self._embed_new_images(missing, progress=progress)
            current.extend(new_rows)
        elif progress:
            progress(1, 1)

        # Keep only files that still exist and valid embeddings.
        current = [r for r in current if Path(r[0]).exists()]
        self._save_index(idx_path, current)

        if not current:
            raise RuntimeError('No readable images were found in this library.')

        valid_paths = [Path(r[0]) for r in current]
        vecs = np.stack([r[2] for r in current]).astype(np.float32, copy=False)
        return valid_paths, vecs, len(missing)

    def text_embeddings(self, texts):
        inp = self.proc(text=texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with self.torch.inference_mode():
            v = self.model.get_text_features(**inp).float().cpu().numpy()
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def match(self, titles, paths, progress=None, topk=5):
        valid_paths, iv, changed_count = self.image_embeddings(paths, progress)
        tv = self.text_embeddings(titles)
        scores = tv @ iv.T

        result = []
        used = {}
        for row in scores:
            order = np.argsort(-row)
            pool = order[:min(len(order), max(50, topk * 10))]
            ranked = sorted(
                pool,
                key=lambda j: -(float(row[j]) - 0.055 * used.get(int(j), 0))
            )
            picks = ranked[:topk]
            if picks:
                used[int(picks[0])] = used.get(int(picks[0]), 0) + 1
            result.append([(valid_paths[int(j)], float(row[j])) for j in picks])

        return result, changed_count, len(valid_paths)
