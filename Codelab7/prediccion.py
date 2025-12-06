
import re, random, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
import joblib

random.seed(42); np.random.seed(42)

positivos = [
    "Excelente servicio","Muy buena atención","Me encantó el producto",
    "Rápido y confiable","Todo llegó perfecto","Calidad superior",
    "Lo recomiendo totalmente","Volveré a comprar","Precio justo y buena calidad",
    "El soporte fue amable","Experiencia increíble","Funcionó mejor de lo esperado",
    "Entregado a tiempo","Muy satisfecho","Cinco estrellas",
    "La comida estaba deliciosa","El empaque impecable","Súper recomendable",
    "Buen trato del personal","Gran experiencia"
]

negativos = [
    "Pésimo servicio","Muy mala atención","Odio este producto",
    "Lento y poco confiable","Llegó dañado","Calidad terrible",
    "No lo recomiendo","No vuelvo a comprar","Caro y mala calidad",
    "El soporte fue grosero","Experiencia horrible","Peor de lo esperado",
    "Entregado tarde","Muy decepcionado","Una estrella",
    "La comida estaba fría","El empaque roto","Nada recomendable",
    "Mal trato del personal","Mala experiencia"
]

def variantes(frase):
    extras = ["", "!", "!!", " 🙂", " 😡", " de verdad", " en serio", " 10/10", " 1/10",
              " súper", " la verdad", " jamás", " nunca", " para nada"]
    return frase + random.choice(extras)

pos = [variantes(p) for _ in range(8) for p in positivos] 
neg = [variantes(n) for _ in range(8) for n in negativos] #
textos = pos + neg 
etiquetas = [1]*len(pos) + [0]*len(neg)

df = pd.DataFrame({"texto": textos, "etiqueta": etiquetas}).sample(frac=1, random_state=42).reset_index(drop=True)


print("Muestras:", df.shape[0], " | Positivos:", df.etiqueta.sum(), " | Negativos:", len(df)-df.etiqueta.sum())

def limpiar(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["texto_clean"] = df["texto"].apply(limpiar)


X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["texto_clean"], df["etiqueta"], test_size=0.2, random_state=42, stratify=df["etiqueta"]
) 

mayoritaria = int(round(y_train.mean()))  
baseline = (y_test == mayoritaria).mean()
print(f"Baseline (clase mayoritaria): {baseline:.3f}") 

vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2)

X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)


clf = LinearSVC(class_weight="balanced", random_state=42)

clf.fit(X_train, y_train) 


pred = clf.predict(X_test) 
acc = accuracy_score(y_test, pred) 
print(f"\nAccuracy en test: {acc:.3f}  |  Mejora vs baseline: {acc - baseline:.3f}\n")
print("Reporte por clase:")
print(classification_report(y_test, pred, digits=3)) 

cm = confusion_matrix(y_test, pred, labels=[0,1]) 
print("\nMatriz de confusión:")
print(pd.DataFrame(cm, index=["Real 0 (neg)", "Real 1 (pos)"], columns=["Pred 0 (neg)", "Pred 1 (pos)"]))

pipe = make_pipeline( 
    TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2),
    LinearSVC(class_weight="balanced", random_state=42)
)
scores = cross_val_score(pipe, df["texto_clean"], df["etiqueta"], cv=5, scoring="f1_macro")  
print(f"\nCV (5-fold) F1_macro: media={scores.mean():.3f}  ±{scores.std():.3f}")

def predecir(textos_nuevos):
    tx = [limpiar(t) for t in textos_nuevos]
    Xn = vectorizer.transform(tx)  
    p = clf.predict(Xn)
    return ["positivo" if i==1 else "negativo" for i in p]

nuevos = [
    "Llegó en buen estado y dentro del tiempo previsto, nada que reprochar.",
    "El producto cumple, pero sinceramente esperaba algo de mejor calidad.",
    "La atención al cliente fue sorprendentemente buena, resolvieron todo rápido.",
    "El envío tardó una eternidad, pensé que nunca iba a llegar.",
    "Muy bonito a la vista, aunque los materiales se sienten algo frágiles.",
    "Por fin un pedido que llega sin problemas, me sorprendió para bien.",
    "La app de seguimiento es un desastre, nunca mostró dónde estaba mi paquete.",
    "En general, funciona bien y hace lo que necesito.",
    "El empaque llegó destrozado, pero por suerte el contenido estaba intacto.",
    "Me gustó más de lo que esperaba, superó mis expectativas.",
    "No entiendo cómo esto puede tener buenas reseñas, no vale la pena.",
    "Me respondieron rápido y solucionaron mi duda sin complicaciones.",
    "El precio está bien, pero el rendimiento es mediocre.",
    "Sorprendentemente resistente, pensé que duraría menos.",
    "La experiencia fue tan mala que no pienso repetir la compra.",
    "Buen producto, aunque el manual es confuso y mal traducido.",
    "El sonido es mejor de lo que imaginé, recomendadísimo.",
    "El servicio técnico fue inútil, no ayudaron en nada.",
    "Lo uso todos los días y hasta ahora funciona perfecto.",
    "No vale lo que cuesta, hay opciones mejores por menos dinero."
]


print("\nPredicciones en textos nuevos:")
for t, etiqueta in zip(nuevos, predecir(nuevos)):
    print(f"- {t}  ->  {etiqueta}")


joblib.dump(vectorizer, "tfidf.joblib")
joblib.dump(clf, "modelo.joblib")
print("\nModelo y vectorizador guardados.")

vec = joblib.load("tfidf.joblib")
model = joblib.load("modelo.joblib")
Xn = vec.transform(["La compra fue excelente, todo perfecto"])
print("Pred loaded model:", "positivo" if model.predict(Xn)[0]==1 else "negativo")