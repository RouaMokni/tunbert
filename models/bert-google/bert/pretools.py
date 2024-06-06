import unicodedata
import json
import re
import pandas as pd
import os
import random

def removeDiacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')

def loadDict(path):
    # Charger le fichier Excel et créer un dictionnaire de traduction
    data = pd.read_excel(path)
    # Replace NaN values with a placeholder or remove rows with NaN in relevant columns
    data = data.dropna(subset=['Column3', 'Column5', 'Column8'])
    data['Column3'].apply(removeDiacritics)
    data['Column5'].apply(removeDiacritics)
    data['Column8'].apply(removeDiacritics)
    arabish_coda_dict = dict(zip(data['Column3'], data['Column5'].astype(str)))
    normm_lemma = dict(zip(data['Column5'].astype(str), data['Column8'].astype(str)))
    return arabish_coda_dict, normm_lemma

def replace_arabic_with_coda(tokens, arabish_coda_dict):
    """Traduire les tokens en utilisant le dictionnaire."""
    new_tokens = []
    for token in tokens:
        token = removeDiacritics(token)
        new_tokens.append(arabish_coda_dict.get(token, token))
    return new_tokens

def normalize_tokens(tokens, normm_lemma):
    """Normaliser les tokens en utilisant le dictionnaire."""
    new_tokens = []
    for token in tokens:
        lemma = normm_lemma.get(token, token).strip()
        if lemma == "PUNC":
            continue
        if lemma == "foreign" or lemma == "NOUN_NUM":
            new_tokens.append(token)
        else:
            new_tokens.append(lemma)
    return new_tokens

def read_documents(file_path):
    discussions = json.load(open(file_path, 'r', encoding='utf-8'))
    
    #for i, discussion in enumerate(discussions):
    #    for j, itturance in enumerate(discussion):
    #        documents[f"{i}_{j}"] = itturance
    return discussions

def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set(set2)))
    union = len(set(set1).union(set(set2)))
    return intersection / union

# Prétraiter la requête pour TunBERT
def pre_query_bert(query, arabish_coda_dict):
    question = removeDiacritics(query)
    question_tokens = re.findall(r"(\w+|[^\s]+)", question)
    return replace_arabic_with_coda(question_tokens, arabish_coda_dict)

# Normaliser la requête pour Jaccard
def norm_query_jaccard(query, normm_lemma, arabish_coda_dict):
    # Normalisation de la question
    question = removeDiacritics(query)
    question_tokens = re.findall(r"(\w+|[^\s]+)", question)
    translated_question_tokens = replace_arabic_with_coda(question_tokens, arabish_coda_dict)
    return normalize_tokens(translated_question_tokens, normm_lemma)

# S1 : nheb ne5edh 9ardh -> hab 5dhe 9ardh
# S2 : nheb ne5ou 9arth -> hab 5dhe 9ardh
def search(prequery, discussions, normm_lemma):
    """Search and return answers of top 5 similar questions."""
    # Search
    results = []
    for disc_id, discussion in enumerate(discussions):
        for utter_id, utterance in enumerate(discussion):
            doc_tokens = normalize_tokens(re.findall(r"(\w+|\S+)", utterance["question_norm"].lower()), normm_lemma)
            similarity = jaccard_similarity(set(prequery), set(doc_tokens))
            results.append((disc_id, utter_id, similarity))
    # Sort results by similarity score in descending order and return top 5
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:3]

def topReponses(prequery, discussions, normm_lemma):
    # Perform the search and get the top 5 results
    search_results = search(prequery, discussions, normm_lemma)
    reponses = []
    for result in search_results:
        disc_id, utter_id, similarity = result
        #if similarity>=0.5:
        reponses.append(discussions[disc_id][utter_id]["reponse"])
    return reponses
