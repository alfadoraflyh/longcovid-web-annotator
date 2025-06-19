# processing.py
import os, json, numpy as np, faiss
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from tqdm import tqdm

# Load environment variables
load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

# Create Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def get_ontology_embeddings_from_neo4j_medical_embedding_hierarchy():
    """Retrieve symptom labels and embeddings from Neo4j."""
    symptoms, embeddings = [], []
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Symptom)
            WHERE s.ME_hierarchical_phrase IS NOT NULL
            RETURN s.label_name AS label, s.ME_hierarchical_phrase AS embedding
        """)
        for record in result:
            label = record["label"]
            embedding = record["embedding"]
            if label in ["symptom", "early symptom", "severe symptom"]:
                continue
            symptoms.append(label)
            embeddings.append(embedding)
    return symptoms, embeddings

def embed_text_sentence_transformer(text: str, model):
    """Generate an embedding vector for the input text using SentenceTransformer."""
    embedding = model.encode([text])
    return embedding[0]  # Return as a 1D numpy array

def build_faiss_index(embeddings):
    """Build a FAISS index from a list of embedding vectors."""
    embedding_array = np.array(embeddings, dtype='float32')
    faiss.normalize_L2(embedding_array)
    dim = embedding_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embedding_array)
    return index

def query_faiss_index(index, user_vector, top_k=5):
    """Query the FAISS index to find top_k similar embeddings."""
    query_vec = user_vector.astype('float32')
    faiss.normalize_L2(query_vec.reshape(1, -1))
    distances, indices = index.search(query_vec.reshape(1, -1), top_k)
    return indices[0], distances[0]

def clean_text(input_text):
    """Extract valid JSON from input text."""
    first_brace_candidates = [input_text.find(x) for x in ['{', '['] if x in input_text]
    if not first_brace_candidates:
        return input_text
    start_index = min(first_brace_candidates)
    end_index = max(input_text.rfind('}'), input_text.rfind(']'))
    if end_index == -1:
        return input_text
    json_text = input_text[start_index:end_index + 1]
    try:
        obj = json.loads(json_text)
        return json.dumps(obj)
    except Exception:
        # Fallback repair mechanism if needed
        return json_text

# Initialize models and indices (executed once during startup)
symptoms, symptom_embeddings = get_ontology_embeddings_from_neo4j_medical_embedding_hierarchy()
faiss_index = build_faiss_index(symptom_embeddings)
EMBEDDING_MODEL = 'abhinand/MedEmbed-base-v0.1'
sentence_transformer_model = SentenceTransformer(EMBEDDING_MODEL)
TOP_K = 10

# Configure LangChain LLM chain
MODEL_GPT = "gpt-4o-mini"
TEMPERATURE = 0.0
llm = ChatOpenAI(model=MODEL_GPT, temperature=TEMPERATURE, api_key=OPEN_API_KEY)

prompt_template = PromptTemplate(
    input_variables=["user_text", "candidate_symptoms"],
    template="""
    You are a precise medical text annotator. You receive:
    1) A user’s text: {user_text}
    2) A list of candidate symptom concepts from a COVID-19 ontology: {candidate_symptoms}

    Your task is to:
    1. Identify each substring in the user's text that indicates a symptom (verbatim).
       - If multiple related symptoms appear in one phrase connected by "and" or "or," treat that entire phrase as a single symptom mention.
       - For each symptom mention:
         - "symptom": the exact substring from the user text (verbatim, including punctuation or special characters).
         - "recognized_concept":
             - If it matches a concept in the provided candidate list, output that concept name.
             - Otherwise, output "undetected".

    2. Identify all other time expressions in the user text, even if they are not clearly linked to a symptom.
       - A valid time expression must explicitly mention a time indicator (e.g., numbers + units like days/weeks/months/years, or words like "yesterday," "later," "after," "since").
       - For each time expression:
         - "time_expression": the exact substring (verbatim) from the user text indicating time or duration.
         - "context":
             - If it's clearly tied to a symptom, mention that symptom’s exact substring here.
             - Otherwise, use "general" if no direct symptom link is apparent.

    3. Output Format:
       - Return **only** a valid JSON object with two top-level keys: "symptoms" and "time_expressions".
         For example:
         {{
           "symptoms": [
             {{
               "symptom": "<string>",
               "recognized_concept": "<string>",
             }},
             ...
           ],
           "time_expressions": [
             {{
               "time_expression": "<string>",
               "context": "<string>"
             }},
             ...
           ]
         }}
       - Do not include any additional text or keys.

    Now process the user text according to these instructions and produce your JSON response.
    """
)

annotation_chain = prompt_template | llm

# For simplicity we define a function that processes one input text.
def process_RAG(user_text: str) -> dict:
    # Tokenize and generate n-grams
    words = word_tokenize(user_text)
    ngram_list = list(ngrams(words, 2))
    all_top_symptoms = set()
    for ngram in ngram_list:
        ngram_text = ' '.join(ngram)
        user_vector = embed_text_sentence_transformer(ngram_text, sentence_transformer_model)
        top_indices, _ = query_faiss_index(faiss_index, user_vector, TOP_K)
        for i in top_indices:
            all_top_symptoms.add(symptoms[i])
    all_top_symptoms.discard('long covid symptom')
    candidate_symptoms = list(all_top_symptoms)
    
    # Prepare chain input and get the annotation
    chain_input = {"user_text": user_text, "candidate_symptoms": candidate_symptoms}
    # Using the chain synchronously for simplicity; in production you might batch inputs
    output = annotation_chain.invoke(chain_input)
    try:
        clean_output = clean_text(output.content)
        return json.loads(clean_output)
    except Exception as e:
        return {"error": f"Processing failed: {e}"}