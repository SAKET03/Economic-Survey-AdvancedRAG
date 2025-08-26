import datetime
import json
import os
import time

from dotenv import load_dotenv
from groq import APIStatusError, RateLimitError
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# ------------------
# 1. Setup
# ------------------
load_dotenv()

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "yourpassword"

# output dir
run_dir = f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(run_dir, exist_ok=True)
output_file = os.path.join(run_dir, "answers.json")

# ------------------
# 2. Connect Vector DB
# ------------------
print("[Status] Initializing embedding + Neo4j vectorstore...")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = Neo4jVector(
    embedding=embed_model,
    url=URI,
    username=USER,
    password=PASSWORD,
    index_name="chunkEmbeddings",
    node_label="Chunk",
    text_node_property="content",
    embedding_node_property="embedding",
    search_type="hybrid",
)

# ------------------
# 3. Load Test Questions
# ------------------
questions = [
    "What is India’s GDP growth projection for FY 2025-26?",
    "What was India’s real GDP growth in FY 2024-25?",
    "What was the CPI inflation rate in FY 2024-25?",
    "Why does the Survey emphasise deregulation for growth?",
    "What was India’s Current Account Deficit in FY 2024-25?",
    "How large were India’s forex reserves as of March 2025?",
    "How does the one view the services sector’s role in GDP?",
    "What is the fiscal deficit target for FY 2025-26?",
    "What about climate shocks and inflation?",
    "What is the stance on AI’s labour market impact?",
    "How do India’s GDP and inflation trends compare in FY 2024-25?",
    "How does fiscal consolidation help inflation management?",
    "What risks does India face due to reliance on Chinese supply chains?",
    "How did FDI inflows in FY 2024-25 compare with FY 2023-24?",
    "What lessons does India draw from Europe’s competitiveness issues?",
    "How is agriculture framed as a sector of the future?",
    "How does energy transition impact India’s competitiveness?",
    "What are the employment and skills policies shaping up ?",
    "How does deregulation affect state-wise industrialisation?",
    "How does health & education policy affect long-term growth?",
    "How do growth, inflation, and fiscal deficit trends interact FY 2023-24 → 2025-26?",
    "Is there any correlation between deregulation, investment, and services performance?",
    "What are long-term risks if India fails to indigenise EV/solar supply chains?",
    "How does the “China challenge” affect India’s manufacturing?",
    "How do fiscal deficit & inflation dynamics shape bond yields?",
    "How are NEP & skilling linked to AI-era employment?",
    "How do external shocks and domestic demand resilience interact?",
    "How do lifestyle factors (processed food, mental health) affect growth?",
    "What is the combined message of fiscal, monetary & external indicators for FY 2025-26?",
    "How India must adapt Western policy lessons?",
]

# ------------------
# 4. Setup Reranker
# ------------------
print("[Status] Loading reranker model...")
r_model = CrossEncoder("BAAI/bge-reranker-base")

# ------------------
# 5. Setup LLM (ChatGroq)
# ------------------
print("[Status] Initializing ChatGroq LLM...")
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
)

prompt_template = """
You are an expert analyst specializing in Indian economic policy and data analysis. You have access to relevant documents from India's Economic Survey and related policy documents. Your task is to provide accurate, comprehensive answers based STRICTLY on the provided context documents.

Question: {question}

Context:
{context}

Answer:
"""

# ------------------
# 6. Main loop with progress
# ------------------
results = []
print("[Status] Starting Q&A pipeline...")
for q in tqdm(questions, desc="Processing questions", ncols=100):
    # Step 1: Retrieve top 25 candidates
    candidates = vectorstore.similarity_search_with_relevance_scores(q, k=25)

    # Step 2: Prepare for reranker
    texts = [doc.page_content for doc, _ in candidates]
    pairs = [(q, t) for t in texts]

    scores = r_model.predict(pairs)

    reranked = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)

    # Step 3: Filter top 1–3 chunks above threshold (store content + metadata)
    top_chunks = []
    for doc, score in reranked:
        if score > 0.6:  # threshold
            top_chunks.append(
                {
                    "content": doc.page_content,  # ✅ actual text
                    "chapter_number": doc.metadata.get("chapter_number"),
                    "subchapter_number": doc.metadata.get("subchapter_number"),
                    "tags": doc.metadata.get("tags"),
                    "score": score,
                }
            )
        if len(top_chunks) == 3:
            break

    if len(top_chunks) == 0 and reranked:
        # force at least 1
        doc, score = reranked[0]
        top_chunks.append(
            {
                "content": doc.page_content,
                "chapter_number": doc.metadata.get("chapter_number"),
                "subchapter_number": doc.metadata.get("subchapter_number"),
                "tags": doc.metadata.get("tags"),
                "score": score,
            }
        )

    # ------------------
    # 4. Prepare context string for LLM (ONLY content)
    # ------------------
    context = "\n\n".join([c["content"] for c in top_chunks])

    # ------------------
    # 5. Query LLM with retry logic
    # ------------------
    formatted_prompt = prompt_template.format(question=q, context=context)
    while True:
        try:
            answer = llm.invoke(formatted_prompt).content
            break
        except RateLimitError:
            print("\n[Warning] Rate limit hit. Waiting 60 seconds before retrying...")
            time.sleep(60)
        except APIStatusError as e:
            print(f"\n[Warning] API status error: {e}.")
            context = context[:29500]
            formatted_prompt = prompt_template.format(question=q, context=context)
        except Exception as e:
            print(f"\n[Error] {e}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Save result (answer + chunks with metadata for post-processing)

    for chunk in top_chunks:
        chunk.pop("content", None)

    results.append({"question": q, "answer": answer, "chunks": top_chunks})

# ------------------
# 7. Save to JSON + Preview
# ------------------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[Status] Saved results to {output_file}")
