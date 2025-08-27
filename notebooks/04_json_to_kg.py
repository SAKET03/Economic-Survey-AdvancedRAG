import json
import os
from typing import Dict, List, Set

from py2neo import Graph, Node, NodeMatcher, Relationship

# Define Neo4j database connection parameters
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "yourpassword"


def create_knowledge_graph_from_chunks(json_directory: str):
    """
    Create a knowledge graph from processed chunk JSON files.

    Args:
        json_directory (str): The directory containing chunk JSON files to be processed.
    """
    # Connect to the Neo4j database
    graph = Graph(URI, auth=(USER, PASSWORD))

    # Delete all existing nodes and relationships in the graph
    # print("Clearing existing graph...")
    # graph.delete_all()

    # Create constraints for better performance and data integrity
    print("Creating constraints...")
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
    graph.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chapter) REQUIRE ch.number IS UNIQUE"
    )
    graph.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Concept) REQUIRE t.name IS UNIQUE"
    )

    # Initialize a NodeMatcher for finding existing nodes
    matcher = NodeMatcher(graph)

    # Track processed entities
    processed_chapters: Set[int] = set()
    processed_concepts: Set[str] = set()
    chunk_count = 0

    # Get all JSON files and sort them for consistent processing
    json_files = [f for f in os.listdir(json_directory) if f.endswith(".json")]
    json_files.sort()  # Ensure consistent ordering

    print(f"Found {len(json_files)} JSON files to process...")

    # Process each JSON file
    for filename in json_files:
        file_path = os.path.join(json_directory, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Extract data from JSON
            chapter_no = data["chapter_no"]
            subchapter_no = data["subchapter_no"]
            content = data["content"]
            tags = data["tags"]

            # Create or get Chapter node
            chapter_node = create_or_get_chapter_node(
                graph, matcher, chapter_no, processed_chapters
            )

            # Create Chunk node
            chunk_node = create_chunk_node(
                graph, chapter_no, subchapter_no, content, tags
            )

            # Create relationship between Chapter and Chunk
            chapter_chunk_rel = Relationship(chapter_node, "CONTAINS", chunk_node)
            graph.create(chapter_chunk_rel)

            # Process tags and create Concept nodes
            for tag in tags:
                concept_node = create_or_get_concept_node(
                    graph, matcher, tag, processed_concepts
                )

                # Create relationship between Chunk and Concept
                chunk_concept_rel = Relationship(chunk_node, "RELATES_TO", concept_node)
                graph.create(chunk_concept_rel)

            chunk_count += 1
            print(
                f"Processed: {filename} -> Chapter {chapter_no}, Chunk {subchapter_no}, Tags: {tags}"
            )

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Create additional relationships between concepts (co-occurrence)
    print("Creating concept co-occurrence relationships...")
    create_concept_cooccurrence_relationships(graph, json_directory)

    # Print summary
    print("\nKnowledge graph creation complete!")
    print(f"- Processed {chunk_count} chunks")
    print(f"- Created {len(processed_chapters)} chapter nodes")
    print(f"- Created {len(processed_concepts)} concept nodes")

    # Print some graph statistics
    print_graph_statistics(graph)


def create_or_get_chapter_node(
    graph: Graph, matcher: NodeMatcher, chapter_no: int, processed_chapters: Set[int]
) -> Node:
    """Create or retrieve a Chapter node"""
    if chapter_no not in processed_chapters:
        chapter_node = Node(
            "Chapter",
            number=chapter_no,
            title=f"Chapter {chapter_no}",
            id=f"chapter_{chapter_no}",
        )
        graph.merge(chapter_node, "Chapter", "number")
        processed_chapters.add(chapter_no)
        print(f"Created Chapter {chapter_no} node")
    else:
        chapter_node = matcher.match("Chapter", number=chapter_no).first()

    return chapter_node


def create_chunk_node(
    graph: Graph, chapter_no: int, subchapter_no: int, content: str, tags: List[str]
) -> Node:
    """Create a Chunk node"""
    # Truncate content for display purposes (keep full content in node)

    chunk_node = Node(
        "Chunk",
        chapter_number=chapter_no,
        subchapter_number=subchapter_no,
        content=content,
        tags=tags,
    )

    graph.create(chunk_node)
    return chunk_node


def create_or_get_concept_node(
    graph: Graph, matcher: NodeMatcher, tag: str, processed_concepts: Set[str]
) -> Node:
    """Create or retrieve a Concept node"""
    if tag not in processed_concepts:
        concept_node = Node(
            "Concept",
            name=tag,
            normalized_name=tag.lower().replace("_", " "),
        )
        graph.merge(concept_node, "Concept", "name")
        processed_concepts.add(tag)
        print(f"Created Concept node: {tag}")
    else:
        concept_node = matcher.match("Concept", name=tag).first()

    return concept_node


def create_concept_cooccurrence_relationships(graph: Graph, json_directory: str):
    """Create relationships between concepts that co-occur in the same chunks"""
    concept_cooccurrence: Dict[str, Set[str]] = {}

    # Collect co-occurrence data
    for filename in os.listdir(json_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(json_directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                tags = data["tags"]

                # For each pair of tags in the same chunk, record co-occurrence
                for i, tag1 in enumerate(tags):
                    for tag2 in tags[i + 1 :]:
                        if tag1 not in concept_cooccurrence:
                            concept_cooccurrence[tag1] = set()
                        if tag2 not in concept_cooccurrence:
                            concept_cooccurrence[tag2] = set()

                        concept_cooccurrence[tag1].add(tag2)
                        concept_cooccurrence[tag2].add(tag1)

            except Exception as e:
                print(f"Error processing co-occurrence for {filename}: {e}")
                continue

    # Create co-occurrence relationships in Neo4j
    matcher = NodeMatcher(graph)
    cooccurrence_count = 0

    for concept1, related_concepts in concept_cooccurrence.items():
        concept1_node = matcher.match("Concept", name=concept1).first()

        if concept1_node:
            for concept2 in related_concepts:
                concept2_node = matcher.match("Concept", name=concept2).first()

                if concept2_node:
                    # Create bidirectional co-occurrence relationship
                    cooccur_rel = Relationship(
                        concept1_node, "CO_OCCURS_WITH", concept2_node
                    )
                    graph.merge(cooccur_rel)
                    cooccurrence_count += 1

    print(f"Created {cooccurrence_count} concept co-occurrence relationships")


def print_graph_statistics(graph: Graph):
    """Print statistics about the created graph"""
    try:
        # Count nodes by type
        chapter_count = graph.run("MATCH (c:Chapter) RETURN count(c) as count").data()[
            0
        ]["count"]
        chunk_count = graph.run("MATCH (c:Chunk) RETURN count(c) as count").data()[0][
            "count"
        ]
        concept_count = graph.run("MATCH (c:Concept) RETURN count(c) as count").data()[
            0
        ]["count"]

        # Count relationships by type
        contains_count = graph.run(
            "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count"
        ).data()[0]["count"]
        relates_count = graph.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
        ).data()[0]["count"]
        cooccur_count = graph.run(
            "MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) as count"
        ).data()[0]["count"]

        print("\nGraph Statistics:")
        print("- Nodes:")
        print(f"  - Chapters: {chapter_count}")
        print(f"  - Chunks: {chunk_count}")
        print(f"  - Concepts: {concept_count}")
        print("- Relationships:")
        print(f"  - CONTAINS (Chapter->Chunk): {contains_count}")
        print(f"  - RELATES_TO (Chunk->Concept): {relates_count}")
        print(f"  - CO_OCCURS_WITH (Concept<->Concept): {cooccur_count}")

        # Show top concepts by frequency
        print("\nTop 10 Most Connected Concepts:")
        top_concepts = graph.run("""
            MATCH (c:Concept)<-[r:RELATES_TO]-()
            RETURN c.name as concept, count(r) as connections
            ORDER BY connections DESC
            LIMIT 10
        """).data()

        for i, concept_data in enumerate(top_concepts, 1):
            print(
                f"  {i}. {concept_data['concept']}: {concept_data['connections']} connections"
            )

    except Exception as e:
        print(f"Error generating statistics: {e}")


def verify_graph_creation(graph: Graph):
    """Verify the graph was created correctly"""
    try:
        # Test queries
        print("\nVerification Queries:")

        # Sample chapters
        chapters = graph.run(
            "MATCH (c:Chapter) RETURN c.number as number LIMIT 5"
        ).data()
        print(f"Sample Chapters: {[ch['number'] for ch in chapters]}")

        # Sample concepts
        concepts = graph.run("MATCH (c:Concept) RETURN c.name as name LIMIT 10").data()
        print(f"Sample Concepts: {[c['name'] for c in concepts]}")

        # Sample chunk with its relationships
        sample_chunk = graph.run("""
            MATCH (ch:Chapter)-[:CONTAINS]->(c:Chunk)-[:RELATES_TO]->(con:Concept)
            RETURN ch.number as chapter, c.subchapter_number as subchapter, 
                   collect(con.name) as concepts
            LIMIT 3
        """).data()

        print("Sample Chunk Relationships:")
        for chunk in sample_chunk:
            print(
                f"  Chapter {chunk['chapter']}, Subchapter {chunk['subchapter']}: {chunk['concepts']}"
            )

    except Exception as e:
        print(f"Error in verification: {e}")


# Main execution function
def main():
    """Main function to execute the knowledge graph creation"""
    # Update this path to your processed chunks directory
    # json_directory = "processed_chunks"
    # json_directory = "/workspace/processed_chunks_24-25"
    # json_directory = "/workspace/processed_chunks_24-25"
    # json_directory = "/workspace/output/03/ES_24-25"
    json_directory = "/workspace/output/03/"

    if not os.path.exists(json_directory):
        print(f"Error: Directory '{json_directory}' does not exist.")
        print(
            "Please update the json_directory path to point to your processed chunks."
        )
        return

    try:
        for i in os.listdir(json_directory):
            # Create the knowledge graph
            create_knowledge_graph_from_chunks(os.path.join(json_directory, i))

            # Verify the creation
            graph = Graph(URI, auth=(USER, PASSWORD))
            verify_graph_creation(graph)

            print(
                f"\nKnowledge graph successfully created from {os.path.join(json_directory, i)}"
            )
            print("You can now query the graph using Neo4j Browser or other tools.")

    except Exception as e:
        print(f"Error creating knowledge graph: {e}")


if __name__ == "__main__":
    main()
