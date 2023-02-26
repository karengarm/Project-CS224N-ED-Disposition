from gensim.models import doc2vec
from lbl2vec import Lbl2Vec
import pandas as pd

keywords = [
	["Pneumonia"],
	["Pneumothorax"],
	["Pleural Effusion"],
	["Pulmonary edema"],
	["Rib fracture"],
	["Infection"],
	["Aspiration"],
	["Cardiomegaly"],
	["Opacities"],
	["Atelectasis"],
	["Intracranial hemorrhage"],
	["Subarachnoid hemorrhage"],
	["Subdural hemorrhage"],
	["Epidural hemorrhage"],
	["Intraparenchymal hemorrhage"],
	["Intraventricular hemorrhage"],
	["Skull fracture"],
	["Stroke"],
	["Cerebral edema"],
	["Diffuse axonal injury"],
	["Appendicitis "],
	["Cholecystitis"],
	["Abdominal Aortic Aneurysm"],
	["Small bowel obstruction"],
	["Pancreatitis"],
	["Splenic laceration"],
	["Liver laceration"],
	["Colitis"],
	["Pyelonephritis"],
	["Nephrolithiasis"],
	["Malignancy"],
	["Pneumonia"],
	["Pneumothorax"],
	["Pleural Effusion"],
	["Pulmonary edema"],
	["Rib fracture"],
	["Pericaridial effusion"],
	["Aortic dissection"],
	["Malignancy"],
]

keywords_small = keywords[:3]

df = pd.read_csv('datasets/ed.csv')
df = df.reset_index()

def str_to_document(index, input_str):
	"""
	convert string into TaggedDocument
	"""
	return doc2vec.TaggedDocument(input_str.split(), tags=[str(index)])


df["doc"] = df.apply(lambda x: str_to_document(x['index'], x['Impression']), axis=1)

# d2v_model = doc2vec.Doc2Vec(documents=df["doc"].tolist(), vector_size=5, window=2, min_count=1, workers=4, dbow_words=1, dm=0)
# l2v_model = Lbl2Vec(keywords_list=keywords_small, doc2vec_model=d2v_model)
l2v_model = Lbl2Vec(
	keywords_list=keywords_small,
	tagged_documents=df["doc"].tolist(),
	min_count=1)

l2v_model.fit()
l2v_model.predict_model_docs()