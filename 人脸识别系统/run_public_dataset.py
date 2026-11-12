"""Reproducible aligned-face benchmark, separate from camera acceptance data."""
import csv
import hashlib
import json
import random
import re
import zipfile
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np

from faceid.models import PCARecognizer, LBPHRecognizer
from faceid.calibration import choose_threshold
from faceid.evaluation import build_metrics


def main():
    root = Path(__file__).resolve().parent
    source = root / 'datasets/extended_yale_b/selected_cropped.zip'
    output = root / 'outputs/public_yale_b'
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    with zipfile.ZipFile(source) as archive:
        for name in sorted(archive.namelist()):
            match = re.search(r'(yaleB\d+)_P00A([+-]\d+)E([+-]\d+)\.pgm$', name, re.I)
            if match:
                raw = archive.read(name)
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    groups[match[1]].append(dict(source=name, raw=raw, image=image,
                        angle=abs(int(match[2]))+abs(int(match[3]))))
    registered = sorted(k for k, v in groups.items() if len(v) >= 60)[:10]
    unknown = sorted(k for k, v in groups.items() if k not in registered and len(v) >= 20)[:4]
    subjects = registered + unknown
    if len(registered) < 10 or len(unknown) < 4:
        raise ValueError('Need 10 registered and 4 unknown subjects: ' + str({k: len(v) for k,v in groups.items()}))
    samples = []
    rng = random.Random(20261112)
    for i, subject in enumerate(subjects[:14]):
        images = sorted(groups[subject], key=lambda x:(x['angle'],x['source']))
        if i < 10:
            interference = images[-10:]
            remaining = images[:-10]
            rng.shuffle(remaining)
            assignments = [('train', remaining[:30]), ('val_known', remaining[30:40]),
                           ('test_normal', remaining[40:50]), ('test_interference', interference)]
            identity = 'P%03d' % (i+1)
        else:
            rng.shuffle(images)
            assignments = [('val_unknown' if i < 12 else 'test_unknown', images[:20])]
            identity = 'U%03d' % (i-9)
        for split, selected in assignments:
            for item in selected:
                samples.append(dict(item, identity=identity, split=split,
                    scene='extreme_illumination' if split=='test_interference' else 'mixed_illumination',
                    sha256=hashlib.sha256(item['raw']).hexdigest(), batch='unverified'))
    assert len(samples)==680
    assert len({s['sha256'] for s in samples})==680, 'Duplicate image bytes'
    fields=['source','identity','split','scene','sha256','batch']
    with (output/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        w.writerows({k:s[k] for k in fields} for s in samples)
    # Use existing recognizers on already aligned faces; no claim of detector performance.
    for s in samples:
        s['face']=cv2.equalizeHist(cv2.resize(s['image'],(64,64)))
    train=[s for s in samples if s['split']=='train']
    predictions=[]
    calibration={}
    for method, model in [('pca',PCARecognizer(n_components=20)),('lbph',LBPHRecognizer())]:
        model.fit([s['face'] for s in train],[s['identity'] for s in train])
        validation={}
        for split in ['val_known','val_unknown']:
            validation[split]=[(s['identity'],*model.predict(s['face'])) for s in samples if s['split']==split]
        choice=choose_threshold(validation['val_known'],validation['val_unknown'])
        calibration[method]=choice
        threshold=choice['threshold']
        model.save(output/(method+'.npz'))
        for s in samples:
            if not s['split'].startswith('test_'): continue
            candidate,distance=model.predict(s['face'])
            accepted=distance<=threshold
            predictions.append(dict(method=method,truth=s['identity'],split=s['split'],scene=s['scene'],
                source=s['source'],candidate=candidate,distance=float(distance),threshold=threshold,
                status='known' if accepted else 'unknown',identity=candidate if accepted else None))
        print(method, 'finished',flush=True)
    result=dict(dataset='Extended Yale B, cropped frontal subset',seed=20261112,
        archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        counts=dict(Counter(s['split'] for s in samples)),
        protocol='Image-disjoint aligned-face recognition; session independence unverified; illumination extremes reserved for interference.',
        limitations=['No camera/detection evaluation','No independent acquisition batch evidence',
            'No pose or glasses/occlusion test','No background test','Normal set includes varied lighting'],
        calibration=calibration,metrics=build_metrics(predictions))
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['metrics'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
