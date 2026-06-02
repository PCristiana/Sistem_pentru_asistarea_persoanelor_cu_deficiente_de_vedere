# Object Detection cu YOLO pentru imagini de tip COCO subset

Acest proiect are ca scop antrenarea, validarea și compararea unor modele YOLO pentru detecția obiectelor în imagini. Proiectul folosește un subset personalizat din COCO, denumit `coco_blind_subset`, și urmărește detectarea unor clase relevante precum persoane, mașini, biciclete, autobuze, semafoare, indicatoare, bănci, hidranți, câini etc.

Au fost testate modele YOLO în două variante principale:

- model antrenat fără augmentare;
- model antrenat cu augmentări aplicate on-the-fly folosind Albumentations.

Scopul comparației este observarea influenței augmentărilor asupra performanței modelului și asupra capacității acestuia de generalizare pe imagini noi.

---

## Clase detectate

Datasetul folosește următoarele clase:

1. `person`
2. `bicycle`
3. `car`
4. `motorcycle`
5. `bus`
6. `truck`
7. `traffic light`
8. `stop sign`
9. `bench`
10. `fire hydrant`
11. `parking meter`
12. `dog`

---

## Tehnologii folosite

- Python
- Ultralytics YOLO
- PyTorch
- Albumentations
- OpenCV
- Pandas
- Matplotlib
- CUDA / GPU pentru antrenare

---

## Structura proiectului

```text
project/
│
├── train_yolo.py
├── train_yolo_albumentations_on_the_fly.py
├── train_yolo26n_albumentations_on_the_fly.py
├── calculare_metrici.py
├── calculare_metrici_yolo26n.py
├── calculeaza_metrici_on_the_fly.py
├── predict_image.py
├── ugmentare_dataset.py
├── test_yolo.py
└── README.md
```

### Descrierea fișierelor principale

| Fișier | Rol |
|---|---|
| `train_yolo.py` | Antrenează modelul YOLO26n fără augmentări personalizate. |
| `train_yolo_albumentations_on_the_fly.py` | Antrenează YOLO11n cu augmentări Albumentations aplicate în timpul antrenării. |
| `train_yolo26n_albumentations_on_the_fly.py` | Antrenează YOLO26n cu augmentări Albumentations aplicate on-the-fly. |
| `calculare_metrici.py` | Calculează și salvează metricile pentru modelul validat. |
| `calculare_metrici_yolo26n.py` | Calculează metricile pentru YOLO26n fără augmentări. |
| `calculeaza_metrici_on_the_fly.py` | Calculează metricile pentru modelul antrenat cu augmentări on-the-fly. |
| `predict_image.py` | Rulează detecția pe o imagine individuală folosind modelul `best.pt`. |
| `ugmentare_dataset.py` | Creează un dataset augmentat offline, salvând imagini și etichete noi. |
| `test_yolo.py` | Verifică dacă modelul YOLO poate fi încărcat corect. |

---

## Dataset

Datasetul este organizat în format YOLO:

```text
coco_blind_subset/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── coco_blind.yaml
```

Fișierul `.yaml` conține calea către dataset și lista claselor utilizate.

Exemplu:

```yaml
path: path/to/coco_blind_subset
train: images/train
val: images/val

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck
  6: traffic light
  7: stop sign
  8: bench
  9: fire hydrant
  10: parking meter
  11: dog
```

---

## Augmentări folosite

Pentru varianta augmentată, au fost folosite transformări Albumentations aplicate în timpul antrenării:

- flip orizontal;
- modificare de luminozitate și contrast;
- modificare hue, saturation și value;
- motion blur;
- zgomot gaussian;
- transformări affine: scalare, translație și rotație.

Augmentările sunt aplicate doar pe setul de antrenare, nu și pe setul de validare. Acest lucru este important deoarece validarea trebuie să fie făcută pe imagini nemodificate, pentru a evalua corect capacitatea modelului de generalizare.

---

## Instalare

1. Clonează repository-ul:

```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

2. Creează un mediu virtual:

```bash
python -m venv venv
```

3. Activează mediul virtual:

Pe Windows:

```bash
venv\Scripts\activate
```

Pe Linux / macOS:

```bash
source venv/bin/activate
```

4. Instalează dependențele:

```bash
pip install ultralytics albumentations opencv-python pandas matplotlib torch torchvision
```

---

## Antrenarea modelului

### YOLO26n fără augmentări

```bash
python train_yolo.py
```

Scriptul antrenează modelul YOLO26n timp de 30 de epoci, folosind imaginile definite în fișierul `coco_blind.yaml`.

Parametrii principali utilizați:

```python
epochs=30
imgsz=640
batch=8
device=0
workers=0
```

### YOLO26n cu augmentări Albumentations

```bash
python train_yolo26n_albumentations_on_the_fly.py
```

În acest caz, augmentările sunt aplicate dinamic în timpul antrenării. Imaginile augmentate nu sunt salvate permanent pe disc, ci sunt generate în timpul procesului de training.

---

## Validare și metrici

Pentru calcularea metricilor finale se poate rula:

```bash
python calculare_metrici_yolo26n.py
```

sau, pentru modelul antrenat cu augmentări:

```bash
python calculeaza_metrici_on_the_fly.py
```

Metricile salvate includ:

- Precision;
- Recall;
- mAP50;
- mAP50-95;
- metrici pe fiecare clasă;
- grafice de validare.

Rezultatele sunt salvate automat în fișiere `.csv` și în folderele generate de Ultralytics.

---

## Explicația metricilor

### Precision

Precision arată câte dintre detecțiile făcute de model sunt corecte.

O valoare mare înseamnă că modelul produce puține detecții false.

### Recall

Recall arată câte dintre obiectele reale au fost găsite de model.

O valoare mare înseamnă că modelul ratează mai puține obiecte.

### mAP50

mAP50 reprezintă media performanței modelului pe toate clasele, folosind un prag IoU de 0.5.

Este o metrică folosită frecvent pentru a evalua calitatea detecțiilor.

### mAP50-95

mAP50-95 este o metrică mai strictă, deoarece calculează performanța pentru mai multe praguri IoU, de la 0.5 până la 0.95.

De obicei, această valoare este mai mică decât mAP50.

---

## Loss-uri urmărite în timpul antrenării

În timpul antrenării sunt salvate și reprezentate grafic următoarele loss-uri:

- `box_loss` – eroarea de localizare a bounding box-urilor;
- `cls_loss` – eroarea de clasificare a obiectelor;
- `dfl_loss` – eroare folosită pentru rafinarea poziției bounding box-urilor.

Sunt urmărite atât valorile de train, cât și cele de validation:

- `train/box_loss`
- `train/cls_loss`
- `train/dfl_loss`
- `val/box_loss`
- `val/cls_loss`
- `val/dfl_loss`

Scăderea loss-urilor de antrenare indică faptul că modelul învață progresiv. Compararea lor cu loss-urile de validare ajută la observarea diferenței dintre învățarea pe datele de antrenare și generalizarea pe date noi.

---

## Predicție pe o imagine nouă

Pentru testarea modelului pe o imagine individuală se folosește:

```bash
python predict_image.py
```

Scriptul încarcă modelul `best.pt` și rulează detecția pe imaginea specificată.

Exemplu de folosire în cod:

```python
results = model(
    source=str(image_path),
    save=True,
    conf=0.3
)
```

Parametrul `conf=0.3` înseamnă că modelul păstrează doar detecțiile cu o încredere mai mare sau egală cu 0.3.

---

## Rezultate

În urma antrenării, au fost comparate modelele YOLO26n fără augmentare și YOLO26n cu augmentare. Comparația a fost realizată folosind metrici precum Precision, Recall, mAP50 și mAP50-95, dar și curbe Precision-Recall.

Modelul augmentat are avantajul că vede în timpul antrenării imagini mai variate, ceea ce poate ajuta la generalizare. Totuși, performanța finală depinde și de numărul de imagini disponibile, distribuția claselor și dificultatea obiectelor detectate.

Clasele mari și vizibile, cum ar fi `bus`, `stop sign` sau `fire hydrant`, tind să fie detectate mai bine. Clasele mici sau greu de separat de fundal, cum ar fi `traffic light`, `bench` sau `truck`, pot avea performanțe mai reduse.

---

## Observații importante

- Căile din scripturi sunt absolute și trebuie modificate în funcție de locația datasetului pe calculatorul utilizatorului.
- Augmentările sunt aplicate doar pe setul de antrenare.
- Setul de validare trebuie păstrat nemodificat pentru evaluare corectă.
- Modelul final folosit pentru inferență este `best.pt`, deoarece acesta corespunde celei mai bune performanțe obținute în timpul antrenării.
- Fișierele `.csv` generate pot fi folosite pentru realizarea graficelor și a comparațiilor dintre modele.

---

## Autor

Proiect realizat în cadrul lucrării de licență, având ca temă utilizarea modelelor YOLO pentru detecția obiectelor în imagini.
