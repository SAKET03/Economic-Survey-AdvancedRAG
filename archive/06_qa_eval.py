# %%
import gc
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import Neo4jVector
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from sentence_transformers import CrossEncoder

load_dotenv()


# %%
# Configuration and Setup
class Config:
    # Neo4j Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "yourpassword"

    # Model Configuration
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    # LLM_MODEL = "llama3-70b-8192"  # Optimal for complex economic reasoning
    LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Optimal for complex economic reasoning

    # Processing Configuration
    RETRIEVAL_K = 25  # Documents to retrieve
    RERANKING_K = 12  # Documents to rerank
    CONTEXT_MIN = 3  # Minimum documents for LLM context
    CONTEXT_MAX = 5  # Maximum documents for LLM context
    RERANKER_THRESHOLD = 0.6  # Logical threshold for quality documents
    RERANKER_BATCH_SIZE = 4  # Batch size for reranking

    # Output Configuration
    OUTPUT_FILE = "qa_results.json"
    DETAILED_OUTPUT_FILE = "qa_results_detailed.json"


# Verify Groq API key is set
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY environment variable not set! Please set it with: export GROQ_API_KEY='your_api_key'"
    )

print("✅ GROQ_API_KEY found in environment")


# %%
def clear_gpu_memory(verbose=True):
    """Comprehensive GPU memory clearing function"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        if verbose:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(
                f"GPU Memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"
            )
    else:
        if verbose:
            print("CUDA not available")


def monitor_gpu_memory(step_name: str):
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        # allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        utilization = (reserved / total) * 100
        print(f"[{step_name}] GPU: {utilization:.1f}% | {reserved:.2f}/{total:.2f} GB")


def extract_metadata_fields(doc) -> Dict[str, Any]:
    """Extract chapter and subchapter information from document metadata"""
    metadata = doc.metadata if hasattr(doc, "metadata") else {}

    return {
        "chunk_id": metadata.get("id", "unknown"),
        "chapter_id": metadata.get("chapter_id", metadata.get("chapterId", "unknown")),
        "chapter_title": metadata.get(
            "chapter_title", metadata.get("chapterTitle", "unknown")
        ),
        "subchapter_id": metadata.get(
            "subchapter_id", metadata.get("subchapterId", "unknown")
        ),
        "subchapter_title": metadata.get(
            "subchapter_title", metadata.get("subchapterTitle", "unknown")
        ),
        "page": metadata.get("page", metadata.get("pageNumber", "unknown")),
        "source": metadata.get("source", metadata.get("sourceFile", "unknown")),
    }


# %%
# Question Dataset
QUESTIONS = [
    # Simple Questions
    {
        "id": 1,
        "difficulty": "Simple",
        "type": "Factual",
        "question": "What is India's GDP growth projection for FY 2025-26?",
    },
    {
        "id": 2,
        "difficulty": "Simple",
        "type": "Factual",
        "question": "What was India's real GDP growth in FY 2024-25?",
    },
    {
        "id": 3,
        "difficulty": "Simple",
        "type": "Factual",
        "question": "What was the CPI inflation rate in FY 2024-25?",
    },
    {
        "id": 4,
        "difficulty": "Simple",
        "type": "Commentary",
        "question": "Why does the Survey emphasise deregulation for growth?",
    },
    {
        "id": 5,
        "difficulty": "Simple",
        "type": "Factual",
        "question": "What was India's Current Account Deficit in FY 2024-25?",
    },
    {
        "id": 6,
        "difficulty": "Simple",
        "type": "Factual",
        "question": "How large were India's forex reserves as of March 2025?",
    },
    {
        "id": 7,
        "difficulty": "Simple",
        "type": "Commentary",
        "question": "How does the one view the services sector's role in GDP?",
    },
    {
        "id": 8,
        "difficulty": "Simple",
        "type": "Factual",
        "question": "What is the fiscal deficit target for FY 2025-26?",
    },
    {
        "id": 9,
        "difficulty": "Simple",
        "type": "Commentary",
        "question": "What about climate shocks and inflation?",
    },
    {
        "id": 10,
        "difficulty": "Simple",
        "type": "Mixed",
        "question": "What is the stance on AI's labour market impact?",
    },
    # Medium Questions
    {
        "id": 11,
        "difficulty": "Medium",
        "type": "Mixed",
        "question": "How do India's GDP and inflation trends compare in FY 2024-25?",
    },
    {
        "id": 12,
        "difficulty": "Medium",
        "type": "Commentary",
        "question": "How does fiscal consolidation help inflation management?",
    },
    {
        "id": 13,
        "difficulty": "Medium",
        "type": "Commentary",
        "question": "What risks does India face due to reliance on Chinese supply chains?",
    },
    {
        "id": 14,
        "difficulty": "Medium",
        "type": "Factual",
        "question": "How did FDI inflows in FY 2024-25 compare with FY 2023-24?",
    },
    {
        "id": 15,
        "difficulty": "Medium",
        "type": "Commentary",
        "question": "What lessons does India draw from Europe's competitiveness issues?",
    },
    {
        "id": 16,
        "difficulty": "Medium",
        "type": "Mixed",
        "question": "How is agriculture framed as a sector of the future?",
    },
    {
        "id": 17,
        "difficulty": "Medium",
        "type": "Commentary",
        "question": "How does energy transition impact India's competitiveness?",
    },
    {
        "id": 18,
        "difficulty": "Medium",
        "type": "Commentary",
        "question": "What are the employment and skills policies shaping up?",
    },
    {
        "id": 19,
        "difficulty": "Medium",
        "type": "Commentary",
        "question": "How does deregulation affect state-wise industrialisation?",
    },
    {
        "id": 20,
        "difficulty": "Medium",
        "type": "Commentary",
        "question": "How does health & education policy affect long-term growth?",
    },
    # Complex Questions
    {
        "id": 21,
        "difficulty": "Complex",
        "type": "Mixed",
        "question": "How do growth, inflation, and fiscal deficit trends interact FY 2023-24 → 2025-26?",
    },
    {
        "id": 22,
        "difficulty": "Complex",
        "type": "Commentary",
        "question": "Is there any correlation between deregulation, investment, and services performance?",
    },
    {
        "id": 23,
        "difficulty": "Complex",
        "type": "Commentary",
        "question": "What are long-term risks if India fails to indigenise EV/solar supply chains?",
    },
    {
        "id": 24,
        "difficulty": "Complex",
        "type": "Commentary",
        "question": 'How does the "China challenge" affect India\'s manufacturing?',
    },
    {
        "id": 25,
        "difficulty": "Complex",
        "type": "Commentary",
        "question": "How do fiscal deficit & inflation dynamics shape bond yields?",
    },
    {
        "id": 26,
        "difficulty": "Complex",
        "type": "Commentary",
        "question": "How are NEP & skilling linked to AI-era employment?",
    },
    {
        "id": 27,
        "difficulty": "Complex",
        "type": "Mixed",
        "question": "How do external shocks and domestic demand resilience interact?",
    },
    {
        "id": 28,
        "difficulty": "Complex",
        "type": "Commentary",
        "question": "How do lifestyle factors (processed food, mental health) affect growth?",
    },
    {
        "id": 29,
        "difficulty": "Complex",
        "type": "Commentary",
        "question": "What is the combined message of fiscal, monetary & external indicators for FY 2025-26?",
    },
    {
        "id": 30,
        "difficulty": "Complex",
        "type": "Commentary",
        "question": "How India must adapt Western policy lessons?",
    },
]

print(f"📋 Loaded {len(QUESTIONS)} questions across 3 difficulty levels")

# Count by difficulty and type
difficulty_counts = {}
type_counts = {}
for q in QUESTIONS:
    difficulty_counts[q["difficulty"]] = difficulty_counts.get(q["difficulty"], 0) + 1
    type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1

print(f"Breakdown by Difficulty: {dict(difficulty_counts)}")
print(f"Breakdown by Type: {dict(type_counts)}")

# %%
# Initialize Models and Connections
print("🚀 Initializing models and connections...")

# Clear GPU memory before starting
clear_gpu_memory()

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name=Config.EMBEDDING_MODEL,
    model_kwargs={
        "device": "cuda",
        "trust_remote_code": True,
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 16,
    },
)

print("✅ Embeddings model initialized")
monitor_gpu_memory("After Embeddings")

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=Config.NEO4J_URI,
    username=Config.NEO4J_USERNAME,
    password=Config.NEO4J_PASSWORD,
)

print("✅ Neo4j connected")
print(f"Graph schema preview:\n{graph.schema[:500]}...")

# Create vector store
chunk_vector_store = Neo4jVector.from_existing_graph(
    embedding=embeddings_model,
    url=Config.NEO4J_URI,
    username=Config.NEO4J_USERNAME,
    password=Config.NEO4J_PASSWORD,
    index_name="chunkEmbeddings",
    node_label="Chunk",
    text_node_properties=["content"],
    embedding_node_property="embedding",
    search_type="hybrid",
)

print("✅ Neo4j vector store connected")

# Initialize Groq LLM with optimal settings for economic Q&A
llm = ChatGroq(
    model=Config.LLM_MODEL,
    temperature=0.1,  # Low temperature for factual accuracy
    max_tokens=3000,  # Sufficient for detailed complex answers
    top_p=0.9,  # Focused but not overly restrictive
)

print(f"✅ Groq LLM ({Config.LLM_MODEL}) initialized")


# %%
class QAPipeline:
    def __init__(self, vector_store, embeddings_model, llm, config):
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.config = config
        self.reranker = None

    def load_reranker(self):
        """Load reranker after clearing embeddings model"""
        print("🔄 Switching from embeddings to reranker model...")

        # Free embeddings model to make room for reranker
        del self.embeddings_model
        clear_gpu_memory()
        monitor_gpu_memory("After Embeddings Cleanup")

        # Load reranker with optimal settings
        self.reranker = CrossEncoder(
            self.config.RERANKER_MODEL,
            device="cuda",
            max_length=512,
        )
        print("✅ Reranker loaded")
        monitor_gpu_memory("After Reranker Load")

    def batch_rerank(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Rerank pairs in batches optimized for GTX 1650"""
        all_scores = []
        # total_batches = (
        #     len(pairs) + self.config.RERANKER_BATCH_SIZE - 1
        # ) // self.config.RERANKER_BATCH_SIZE

        for i in range(0, len(pairs), self.config.RERANKER_BATCH_SIZE):
            batch_pairs = pairs[i : i + self.config.RERANKER_BATCH_SIZE]
            batch_scores = self.reranker.predict(batch_pairs)
            all_scores.extend(
                batch_scores.tolist()
                if hasattr(batch_scores, "tolist")
                else batch_scores
            )
            torch.cuda.empty_cache()

        return all_scores

    def select_context_documents(
        self, reranked_docs: List[Tuple[Any, float]]
    ) -> List[Tuple[Any, float]]:
        """
        Select 3-5 documents based on logical threshold, ensuring minimum 3
        """
        # Always include top 3 documents
        selected_docs = reranked_docs[: self.config.CONTEXT_MIN]

        # Add additional documents if they meet the quality threshold
        for doc, score in reranked_docs[
            self.config.CONTEXT_MIN : self.config.CONTEXT_MAX
        ]:
            if score >= self.config.RERANKER_THRESHOLD:
                selected_docs.append((doc, score))
            else:
                break  # Stop at first document below threshold

        return selected_docs

    def retrieve_and_rerank(self, query: str) -> List[Tuple[Any, float]]:
        """Retrieve and rerank documents for a query"""
        # Retrieve documents with similarity search
        sim_results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=self.config.RETRIEVAL_K, score_threshold=0.7
        )

        if not sim_results:
            print("⚠️  No documents found above similarity threshold")
            return []

        print(f"📄 Retrieved {len(sim_results)} documents from vector search")

        # Take top results for reranking
        top_results = sim_results[: self.config.RERANKING_K]

        # Create pairs and rerank
        pairs = [(query, doc.page_content) for doc, _ in top_results]
        scores = self.batch_rerank(pairs)

        # Sort by reranker scores (highest first)
        reranked = sorted(
            zip([doc for doc, _ in top_results], scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Select optimal context documents using threshold logic
        context_docs = self.select_context_documents(reranked)

        print(
            f"🎯 Selected {len(context_docs)} context documents (scores: {[f'{score:.3f}' for _, score in context_docs]})"
        )

        return context_docs

    def generate_system_prompt(self, question_type: str, difficulty: str) -> str:
        """Generate appropriate system prompt based on question characteristics"""
        base_prompt = """You are an expert analyst specializing in Indian economic policy and data analysis. You have access to relevant documents from India's Economic Survey and related policy documents.

Your task is to provide accurate, comprehensive answers based STRICTLY on the provided context documents. You must cite specific chapters/sections when referencing information."""

        if question_type == "Factual":
            specific_prompt = """
FACTUAL QUESTION GUIDELINES:
- Provide precise numerical data, dates, and specific facts
- Quote exact figures when available in the documents
- If multiple years/periods are mentioned, be specific about timeframes
- Reference the specific chapter/section where data is found
- State clearly if exact data is not available in the provided context"""

        elif question_type == "Commentary":
            specific_prompt = """
COMMENTARY QUESTION GUIDELINES:
- Explain the reasoning, analysis, and policy perspectives presented in the documents
- Synthesize insights from multiple parts of the documents when relevant
- Highlight key arguments and their supporting evidence
- Explain cause-and-effect relationships and policy implications
- Reference specific chapters/sections that support your analysis"""

        else:  # Mixed
            specific_prompt = """
MIXED QUESTION GUIDELINES:
- Combine factual data with analytical insights from the documents
- Start with relevant facts and figures, then provide commentary
- Show how data supports or relates to policy discussions in the documents
- Balance quantitative information with qualitative analysis
- Reference specific chapters/sections for both data and analysis"""

        difficulty_guidance = {
            "Simple": "Provide a clear, direct answer focusing on the key point from the documents.",
            "Medium": "Provide a comprehensive answer covering multiple relevant aspects found in the documents.",
            "Complex": "Provide a detailed, multi-faceted analysis showing interconnections between different factors across the documents.",
        }

        return f"""{base_prompt}

{specific_prompt}

DIFFICULTY LEVEL - {difficulty.upper()}:
{difficulty_guidance[difficulty]}

CRITICAL CONSTRAINTS:
- Base your answer ONLY on the provided context documents
- When citing information, mention the relevant chapter/section if available
- If information is not available in the context, state this clearly
- Do not speculate or add information from outside the provided context
- Maintain objectivity and provide balanced analysis based on the documents
- For complex questions, show how different aspects connect across the documents"""

    def generate_answer(
        self, question: Dict[str, Any], context_docs: List[Tuple[Any, float]]
    ) -> Dict[str, Any]:
        """Generate answer using LLM with retrieved context"""
        start_time = time.time()

        # Prepare context with metadata information
        context_sections = []
        sources_used = []

        for i, (doc, score) in enumerate(context_docs):
            metadata = extract_metadata_fields(doc)

            # Add to sources list
            sources_used.append(
                {
                    "doc_index": i + 1,
                    "reranker_score": round(score, 4),
                    "chunk_id": metadata["chunk_id"],
                    "chapter_id": metadata["chapter_id"],
                    "chapter_title": metadata["chapter_title"],
                    "subchapter_id": metadata["subchapter_id"],
                    "subchapter_title": metadata["subchapter_title"],
                    "page": metadata["page"],
                    "source": metadata["source"],
                }
            )

            # Create context section with metadata
            context_header = f"DOCUMENT {i + 1} [Chapter {metadata['chapter_id']}: {metadata['chapter_title']}"
            if metadata["subchapter_id"] != "unknown":
                context_header += f" | Section {metadata['subchapter_id']}: {metadata['subchapter_title']}"
            if metadata["page"] != "unknown":
                context_header += f" | Page {metadata['page']}"
            context_header += f" | Relevance Score: {score:.3f}]:"

            context_sections.append(f"{context_header}\n{doc.page_content}")

        context_text = "\n\n" + "=" * 80 + "\n\n".join(context_sections)

        # Generate system prompt
        system_prompt = self.generate_system_prompt(
            question["type"], question["difficulty"]
        )

        # Create user prompt
        user_prompt = f"""CONTEXT DOCUMENTS:
{context_text}

QUESTION: {question["question"]}

Please provide a comprehensive answer based on the context documents provided above. When referencing information, mention the relevant chapter or section when possible."""

        try:
            # Generate response using Groq
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            answer = response.content.strip()

            processing_time = time.time() - start_time

            return {
                "question_id": question["id"],
                "question": question["question"],
                "difficulty": question["difficulty"],
                "type": question["type"],
                "answer": answer,
                "sources_used": sources_used,
                "context_docs_count": len(context_docs),
                "avg_relevance_score": round(
                    sum(score for _, score in context_docs) / len(context_docs), 4
                ),
                "processing_time_seconds": round(processing_time, 2),
                "status": "success",
            }

        except Exception as e:
            return {
                "question_id": question["id"],
                "question": question["question"],
                "difficulty": question["difficulty"],
                "type": question["type"],
                "answer": f"Error generating answer: {str(e)}",
                "sources_used": sources_used,
                "context_docs_count": len(context_docs),
                "processing_time_seconds": time.time() - start_time,
                "status": "error",
            }

    def process_all_questions(self) -> List[Dict[str, Any]]:
        """Process all questions and generate answers"""
        results = []

        # Load reranker after clearing embeddings
        self.load_reranker()

        print(f"\n🔄 Processing {len(QUESTIONS)} questions with {Config.LLM_MODEL}...")
        print(
            f"Context selection: {Config.CONTEXT_MIN}-{Config.CONTEXT_MAX} docs (threshold: {Config.RERANKER_THRESHOLD})"
        )

        total_start_time = time.time()

        for i, question in enumerate(QUESTIONS, 1):
            print(f"\n{'=' * 80}")
            print(f"📝 Question {i}/{len(QUESTIONS)} (ID: {question['id']})")
            print(f"🏷️  {question['difficulty']} | {question['type']}")
            print(f"❓ {question['question']}")

            try:
                # Retrieve and rerank documents
                context_docs = self.retrieve_and_rerank(question["question"])

                if not context_docs:
                    print("⚠️  No relevant documents found")
                    results.append(
                        {
                            "question_id": question["id"],
                            "question": question["question"],
                            "difficulty": question["difficulty"],
                            "type": question["type"],
                            "answer": "No relevant documents found in the knowledge base for this question.",
                            "sources_used": [],
                            "context_docs_count": 0,
                            "avg_relevance_score": 0,
                            "processing_time_seconds": 0,
                            "status": "no_context",
                        }
                    )
                    continue

                # Generate answer
                result = self.generate_answer(question, context_docs)
                results.append(result)

                print(f"✅ Completed in {result['processing_time_seconds']}s")
                print(
                    f"📊 Used {result['context_docs_count']} docs (avg relevance: {result['avg_relevance_score']:.3f})"
                )
                print(f"💡 Answer preview: {result['answer'][:200]}...")

            except Exception as e:
                print(f"❌ Error processing question {question['id']}: {str(e)}")
                results.append(
                    {
                        "question_id": question["id"],
                        "question": question["question"],
                        "difficulty": question["difficulty"],
                        "type": question["type"],
                        "answer": f"Error processing question: {str(e)}",
                        "sources_used": [],
                        "context_docs_count": 0,
                        "avg_relevance_score": 0,
                        "processing_time_seconds": 0,
                        "status": "error",
                    }
                )

        total_time = time.time() - total_start_time
        print(f"\n🏁 All questions processed in {total_time:.2f} seconds")

        return results


# %%
# Initialize and run QA pipeline
pipeline = QAPipeline(chunk_vector_store, embeddings_model, llm, Config)

# Process all questions
print("🚀 Starting QA Pipeline...")
results = pipeline.process_all_questions()

# %%
# Save results in multiple formats
print("\n💾 Saving results to files...")

# Basic Q&A format for applications
basic_results = []
for result in results:
    basic_qa = {
        "question_id": result["question_id"],
        "question": result["question"],
        "answer": result["answer"],
    }

    # Add source chapters/sections metadata to basic format
    if result.get("sources_used"):
        chapters_used = list(
            set(
                [
                    src["chapter_id"]
                    for src in result["sources_used"]
                    if src["chapter_id"] != "unknown"
                ]
            )
        )
        sections_used = list(
            set(
                [
                    src["subchapter_id"]
                    for src in result["sources_used"]
                    if src["subchapter_id"] != "unknown"
                ]
            )
        )

        basic_qa["chapters_referenced"] = chapters_used
        basic_qa["sections_referenced"] = sections_used

    basic_results.append(basic_qa)

# Detailed format with comprehensive metadata
detailed_results = {
    "metadata": {
        "total_questions": len(results),
        "processing_date": datetime.now().isoformat(),
        "config": {
            "embedding_model": Config.EMBEDDING_MODEL,
            "reranker_model": Config.RERANKER_MODEL,
            "llm_model": Config.LLM_MODEL,
            "retrieval_k": Config.RETRIEVAL_K,
            "reranking_k": Config.RERANKING_K,
            "context_min": Config.CONTEXT_MIN,
            "context_max": Config.CONTEXT_MAX,
            "reranker_threshold": Config.RERANKER_THRESHOLD,
        },
        "performance_stats": {
            "success_rate": round(
                len([r for r in results if r["status"] == "success"])
                / len(results)
                * 100,
                1,
            ),
            "avg_processing_time": round(
                sum(r.get("processing_time_seconds", 0) for r in results)
                / len(results),
                2,
            ),
            "avg_context_docs": round(
                sum(r.get("context_docs_count", 0) for r in results) / len(results), 1
            ),
            "avg_relevance_score": round(
                sum(
                    r.get("avg_relevance_score", 0)
                    for r in results
                    if r.get("avg_relevance_score", 0) > 0
                )
                / len([r for r in results if r.get("avg_relevance_score", 0) > 0]),
                3,
            ),
        },
    },
    "results": results,
}

# Save basic format
with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(basic_results, f, ensure_ascii=False, indent=2)

# Save detailed format
with open(Config.DETAILED_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(detailed_results, f, ensure_ascii=False, indent=2)

print(f"✅ Basic Q&A results saved to: {Config.OUTPUT_FILE}")
print(f"✅ Detailed results with metadata saved to: {Config.DETAILED_OUTPUT_FILE}")

# %%
# Generate comprehensive summary statistics
print("\n📊 COMPREHENSIVE PROCESSING SUMMARY:")
print("=" * 80)

total_questions = len(results)
successful = len([r for r in results if r["status"] == "success"])
errors = len([r for r in results if r["status"] == "error"])
no_context = len([r for r in results if r["status"] == "no_context"])

print("📈 OVERALL PERFORMANCE:")
print(f"   Total Questions Processed: {total_questions}")
print(
    f"   Successful Answers: {successful} ({successful / total_questions * 100:.1f}%)"
)
print(f"   Errors: {errors} ({errors / total_questions * 100:.1f}%)")
print(f"   No Context Found: {no_context} ({no_context / total_questions * 100:.1f}%)")

# Stats by difficulty
print("\n📊 PERFORMANCE BY DIFFICULTY:")
for difficulty in ["Simple", "Medium", "Complex"]:
    diff_results = [r for r in results if r["difficulty"] == difficulty]
    diff_success = len([r for r in diff_results if r["status"] == "success"])
    avg_time = sum(
        r.get("processing_time_seconds", 0)
        for r in diff_results
        if r["status"] == "success"
    )
    avg_time = avg_time / diff_success if diff_success > 0 else 0
    avg_docs = sum(
        r.get("context_docs_count", 0) for r in diff_results if r["status"] == "success"
    )
    avg_docs = avg_docs / diff_success if diff_success > 0 else 0

    print(
        f"   {difficulty:8}: {diff_success:2}/{len(diff_results):2} successful | "
        f"Avg time: {avg_time:.1f}s | Avg docs: {avg_docs:.1f}"
    )

# Stats by type
print("\n📊 PERFORMANCE BY TYPE:")
for qtype in ["Factual", "Commentary", "Mixed"]:
    type_results = [r for r in results if r["type"] == qtype]
    type_success = len([r for r in type_results if r["status"] == "success"])
    if type_results:
        print(f"   {qtype:11}: {type_success:2}/{len(type_results):2} successful")

# Processing time analysis
successful_results = [r for r in results if r["status"] == "success"]
if successful_results:
    times = [r["processing_time_seconds"] for r in successful_results]
    total_time = sum(r.get("processing_time_seconds", 0) for r in results)

    print("\n⏱️  PROCESSING TIME ANALYSIS:")
    print(f"   Total Processing Time: {total_time:.1f} seconds")
    print(f"   Average per Question: {sum(times) / len(times):.1f} seconds")
    print(f"   Fastest Question: {min(times):.1f} seconds")
    print(f"   Slowest Question: {max(times):.1f} seconds")

# Context usage analysis
context_counts = [r.get("context_docs_count", 0) for r in successful_results]
if context_counts:
    print("\n📄 CONTEXT USAGE ANALYSIS:")
    print(
        f"   Average Documents per Answer: {sum(context_counts) / len(context_counts):.1f}"
    )
    print(f"   Min Documents Used: {min(context_counts)}")
    print(f"   Max Documents Used: {max(context_counts)}")

    # Context distribution
    from collections import Counter

    context_dist = Counter(context_counts)
    print(f"   Distribution: {dict(context_dist)}")

# Quality analysis
relevance_scores = [
    r.get("avg_relevance_score", 0)
    for r in successful_results
    if r.get("avg_relevance_score", 0) > 0
]
if relevance_scores:
    print("\n🎯 RELEVANCE QUALITY ANALYSIS:")
    print(
        f"   Average Relevance Score: {sum(relevance_scores) / len(relevance_scores):.3f}"
    )
    print(f"   Highest Relevance: {max(relevance_scores):.3f}")
    print(f"   Lowest Relevance: {min(relevance_scores):.3f}")

# Chapter/section coverage analysis
all_chapters = set()
all_sections = set()
for result in successful_results:
    if result.get("sources_used"):
        for source in result["sources_used"]:
            if source["chapter_id"] != "unknown":
                all_chapters.add(source["chapter_id"])
            if source["subchapter_id"] != "unknown":
                all_sections.add(source["subchapter_id"])

print("\n📚 KNOWLEDGE BASE COVERAGE:")
print(f"   Unique Chapters Referenced: {len(all_chapters)}")
print(f"   Unique Sections Referenced: {len(all_sections)}")
if all_chapters:
    print(f"   Chapters Used: {sorted(list(all_chapters))}")

# %%
# Display sample results with enhanced formatting
print("\n📋 SAMPLE RESULTS SHOWCASE:")
print("=" * 100)

# Show one example from each difficulty level
sample_difficulties = ["Simple", "Medium", "Complex"]
for difficulty in sample_difficulties:
    sample = next(
        (
            r
            for r in results
            if r["difficulty"] == difficulty and r["status"] == "success"
        ),
        None,
    )
    if sample:
        print(f"\n🏷️  {difficulty.upper()} QUESTION EXAMPLE:")
        print("-" * 60)
        print(f"Q{sample['question_id']}: {sample['question']}")
        print(f"\nAnswer: {sample['answer'][:400]}...")

        if sample.get("sources_used"):
            print(f"\n📚 Sources Used ({len(sample['sources_used'])} documents):")
            for i, source in enumerate(
                sample["sources_used"][:3], 1
            ):  # Show first 3 sources
                print(
                    f"   {i}. Chapter {source['chapter_id']}: {source['chapter_title']}"
                )
                if source["subchapter_id"] != "unknown":
                    print(
                        f"      Section {source['subchapter_id']}: {source['subchapter_title']}"
                    )
                print(f"      Relevance: {source['reranker_score']:.3f}")

        print(
            f"\n⏱️  Processing Time: {sample['processing_time_seconds']}s | Relevance: {sample['avg_relevance_score']:.3f}"
        )
        print("=" * 100)

# %%
# Create analysis dashboard data
dashboard_data = {
    "processing_summary": {
        "total_questions": total_questions,
        "successful_answers": successful,
        "error_count": errors,
        "no_context_count": no_context,
        "success_rate_percentage": round(successful / total_questions * 100, 1),
    },
    "performance_by_difficulty": {},
    "performance_by_type": {},
    "processing_times": {
        "total_seconds": round(
            sum(r.get("processing_time_seconds", 0) for r in results), 1
        ),
        "average_per_question": round(
            sum(r.get("processing_time_seconds", 0) for r in successful_results)
            / len(successful_results),
            2,
        )
        if successful_results
        else 0,
        "fastest_question": round(
            min([r.get("processing_time_seconds", 0) for r in successful_results]), 2
        )
        if successful_results
        else 0,
        "slowest_question": round(
            max([r.get("processing_time_seconds", 0) for r in successful_results]), 2
        )
        if successful_results
        else 0,
    },
    "context_analysis": {
        "average_documents_used": round(sum(context_counts) / len(context_counts), 1)
        if context_counts
        else 0,
        "min_documents": min(context_counts) if context_counts else 0,
        "max_documents": max(context_counts) if context_counts else 0,
        "distribution": dict(Counter(context_counts)) if context_counts else {},
    },
    "quality_metrics": {
        "average_relevance_score": round(
            sum(relevance_scores) / len(relevance_scores), 3
        )
        if relevance_scores
        else 0,
        "highest_relevance": round(max(relevance_scores), 3) if relevance_scores else 0,
        "lowest_relevance": round(min(relevance_scores), 3) if relevance_scores else 0,
    },
    "knowledge_coverage": {
        "chapters_referenced": len(all_chapters),
        "sections_referenced": len(all_sections),
        "chapter_list": sorted(list(all_chapters)),
    },
}

# Fill performance by difficulty and type
for difficulty in ["Simple", "Medium", "Complex"]:
    diff_results = [r for r in results if r["difficulty"] == difficulty]
    diff_success = len([r for r in diff_results if r["status"] == "success"])
    dashboard_data["performance_by_difficulty"][difficulty] = {
        "total": len(diff_results),
        "successful": diff_success,
        "success_rate": round(diff_success / len(diff_results) * 100, 1)
        if diff_results
        else 0,
    }

for qtype in ["Factual", "Commentary", "Mixed"]:
    type_results = [r for r in results if r["type"] == qtype]
    type_success = len([r for r in type_results if r["status"] == "success"])
    dashboard_data["performance_by_type"][qtype] = {
        "total": len(type_results),
        "successful": type_success,
        "success_rate": round(type_success / len(type_results) * 100, 1)
        if type_results
        else 0,
    }

# Save dashboard data
dashboard_file = "qa_dashboard_data.json"
with open(dashboard_file, "w", encoding="utf-8") as f:
    json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

print(f"\n📊 Dashboard data saved to: {dashboard_file}")

# %%
# Final summary and recommendations
print("\n🎯 PIPELINE PERFORMANCE SUMMARY:")
print("=" * 80)

if successful >= total_questions * 0.9:
    status_emoji = "🟢"
    status_text = "EXCELLENT"
elif successful >= total_questions * 0.8:
    status_emoji = "🟡"
    status_text = "GOOD"
else:
    status_emoji = "🔴"
    status_text = "NEEDS IMPROVEMENT"

print(f"{status_emoji} Overall Status: {status_text}")
print(
    f"📈 Success Rate: {successful}/{total_questions} ({successful / total_questions * 100:.1f}%)"
)

if successful_results:
    avg_time = sum(r["processing_time_seconds"] for r in successful_results) / len(
        successful_results
    )
    avg_relevance = sum(
        r.get("avg_relevance_score", 0) for r in successful_results
    ) / len(successful_results)
    avg_docs = sum(r.get("context_docs_count", 0) for r in successful_results) / len(
        successful_results
    )

    print(f"⏱️  Average Processing: {avg_time:.1f} seconds per question")
    print(f"📄 Average Context: {avg_docs:.1f} documents per answer")
    print(f"🎯 Average Relevance: {avg_relevance:.3f}")

print("\n📁 OUTPUT FILES CREATED:")
print(f"   • {Config.OUTPUT_FILE} - Clean Q&A format with chapter references")
print(f"   • {Config.DETAILED_OUTPUT_FILE} - Complete results with metadata")
print(f"   • {dashboard_file} - Performance dashboard data")

print("\n🚀 QA Pipeline completed successfully!")
print(f"   Model: {Config.LLM_MODEL}")
print(
    f"   Context Strategy: {Config.CONTEXT_MIN}-{Config.CONTEXT_MAX} docs (threshold: {Config.RERANKER_THRESHOLD})"
)
print(
    f"   Processing completed with {successful} successful answers out of {total_questions} questions"
)


# %%
# Optional: Cleanup GPU memory
def cleanup_pipeline():
    """Clean up all models and free GPU memory"""
    global pipeline

    print("\n🧹 Cleaning up pipeline...")

    if hasattr(pipeline, "reranker") and pipeline.reranker:
        del pipeline.reranker

    clear_gpu_memory()
    monitor_gpu_memory("Final Cleanup")
    print("✅ Pipeline cleanup completed - GPU memory freed")


# Uncomment to cleanup GPU memory
# cleanup_pipeline()

print("\n" + "=" * 80)
print("🏁 NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("=" * 80)
