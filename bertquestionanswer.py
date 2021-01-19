import json
import pdb
from glob import glob

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time


DATASET_PATH = "./comm_use_subset/*.json"

data_dict = dict()
text_list = list()

for json_file in glob(DATASET_PATH):
    with open(json_file) as file:
        data = json.load(file)

      
        if len(data["metadata"]["title"]) != 0:
            value = data["metadata"]["title"]
        else:
            value = data["paper_id"]

        if len(data["abstract"]) != 0:
            key = data["abstract"][0]["text"]
        else:
            key = data["body_text"][0]["text"]

        text_list.append(key)
        data_dict[key] = value


# We use the Bi-Encoder to encode all text,
# so that we can use it with semantic search
bi_encoder = SentenceTransformer('msmarco-distilbert-base-v2')
#bi_encoder = SentenceTransformer('bert-base-nli-mean-tokens')

# Number of text we want to retrieve with the bi-encoder
top_k = 100

# The bi-encoder will retrieve 100 documents.
# We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

# Now we encode all text we have in our COVID 19 Pdfs corpus
corpus_embeddings = bi_encoder.encode(text_list, show_progress_bar=True)


while True:
    query = input("Please enter a question: ")

    # Encode the query using the bi-encoder and find potentially relevant passages
    start_time = time.time()
    question_embedding = bi_encoder.encode(query, show_progress_bar=True, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, text_list[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    end_time = time.time()

    # Output of top-5 hits
    print("Input question:", query)
    print("Results (after {:.3f} seconds):".format(end_time - start_time))
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], text_list[hit['corpus_id']]))
        print("The retrieved title of the paper related to a given question is: " + data_dict[text_list[hit['corpus_id']]]) 

    print("\n\n========\n")