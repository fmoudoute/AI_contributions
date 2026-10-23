# The Making of RAG: Chunking

## Introduction 
Chunking is frequently underemphasized in broader discussions, yet it constitutes a foundational stage in a RAG 
pipeline and plays a decisive role in extraction quality. It refers to the segmentation of source material into 
discrete textual units, or “chunks,” for subsequent processing. Although typically introduced as an early 
preprocessing step, chunking has implications throughout the pipeline, affecting query interpretation, information 
retrieval, and content generation. Inadequate chunking can significantly impair the performance of a RAG system, 
even when the underlying architecture incorporates sophisticated mechanisms intended to enhance response quality 
and relevance. In this article, I will draw on the best of my knowledge to review the most commonly used chunking 
methods, as a foundation for the next articles in this series, The Making of RAG.

## Chunking, or the sacred ingredient
I like to compare chunking in RAG to the foundation of a building. There is no single recommended, indicated, 
or suggested design that fits all scenarios. Depending on the underlying terrain, the load the structure must bear, 
and the judgment of the engineer, one may choose different foundation depths and configurations. The same principle 
applies to chunking, and just as with foundations, poor chunking can have a serious impact on a RAG system, 
potentially rendering the results entirely irrelevant. There are four well-documented impacts of chunking quality 
on a RAG pipeline:
### 1- Accuracy
The granularity and logic of chunks directly determines whether the retrieval module can locate the relevant information for 
a given query. If chunks are too large, they introduce noise that might dilutes the relevance of retrieved content and 
reduces answer precision. If chunks are too small, they may lack sufficient context, leading to incomplete or 
ambiguous answers that omit key facts. More importantly, over time, inaccurate chunking propagates errors through the generation step, 
causing the system to produce responses that are just factually weak or inconsistent.
### 2 - Cost 
Chunking size influences the number of tokens that flow through the retrieval and generation stages, which directly 
affects compute and API costs. Smaller chunks may increase the number of retrieval calls and queries required to 
cover the same information, raising per-request expenses. Larger chunks reduce the number of retrieval operations 
but increase token consumption in the generation phase, which can also be expensive. To summarize, we can safely say that 
an unoptimized chunking strategy can therefore inflate operational costs without improving, and often while reducing, the quality of the 
output - having said that, be careful in getting lost in the chunking optimization matrix!
### 3 - Context Preservation 
One of the biggest challenge of chunking, is to be able to preserve sufficient semantic context for the model to understand relationships between 
concepts within a chunk. If boundaries cut through sentences, clauses, or logical units, the model might lose the 
connections needed to interpret meaning correctly. This disruption could lead to retrieval that is technically correct 
but semantically fragmented, weakening the coherence of the final answer. By contrast, proper chunk boundaries, 
maintain the internal logic of the text and support more faithful generation.
### 4 - Process Efficiency 
Chunking determines how efficiently the pipeline can index, search, and assemble information across the various steps. 
Poorly chosen chunk sizes can lead to slower indexing, higher latency during retrieval, and redundant processing of 
overlapping content. By contrast, well-structured chunks enable faster similarity search, reduce the number of candidate passages, 
and streamline the assembly of context for the model. Consequently, chunking quality has a direct impact on the 
throughput and responsiveness of the entire RAG system.

Chunking will have an impact on the quality of RAG within these 4 angles, but where things become interesting, is that the industry
developed several methods of chunking that would more or less impact some or all of the four angles. Next, lets have
a look at the different methods of chunking:

## Different methods of chunking
### Fixed-Size Chunking
Fixed-size chunking is the baseline method for segmenting text. It operates by splitting the input into chunks of N 
tokens, regardless of semantic meaning or structure. In this approach, the primary objective is to minimize indexing 
complexity and cost. A real-world example would be taking a 100-page report and cutting it into fixed blocks of 50 
tokens, repeating this process every 50 tokens without regard for sentence boundaries or logical units.

Let us examine a concrete example using 3-token fixed-size chunking:

```python
def fixed_size_chunk(text, chunk_size):
    """
    Split text into fixed-size chunks of `chunk_size` tokens (words).
    """
    # We tokenize by splitting on whitespace
    tokens = text.split()
    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks

# Our example
text = "The chairman is considering investing in a new office in the GBA area."
chunk_size = 3

result = fixed_size_chunk(text, chunk_size)

for idx, chunk in enumerate(result, 1):
    print(f"{idx}. {chunk}")
```
#### After:
```text
1. The chairman is
2. considering investing in
3. a new office
4. in the GBA
5. area.
```

### Recursive Chunking

Recursive chunking is a more structured approach that tries to respect the natural organization of text. It operates by splitting 
the input at semantically meaningful boundaries in a hierarchical manner: first by the headings, then by the paragraphs, and 
finally the sentences only when necessary. In the recursive chunking approach, the primary objective is to preserve context and logical 
structure (or the flow...) while maintaining manageable chunk sizes. A real-world example would be taking a 100-page technical report and 
first dividing it by section headings, then by paragraphs within each section, and only breaking down into sentences if 
a paragraph exceeds the target chunk size.

Let us examine a concrete example using recursive chunking on a longer document that still include the same sentence from the 
fixed-size example:

#### Python Code: Recursive Chunking (no external libraries)

```python
def recursive_chunk(text, max_chunk_size=100):
    """
    Split text recursively by:
    1. Headings (lines ending with newline and not too long)
    2. Paragraphs (blocks separated by double newline)
    3. Sentences (only if paragraph exceeds max_chunk_size)
    """
    def split_by_sentences(text_block):
        """Split a block into sentences by period, question mark, or exclamation."""
        sentences = []
        current = []
        for char in text_block:
            current.append(char)
            if char in ['.', '!', '?']:
                sentences.append("".join(current).strip())
                current = []
        if current:
            sentences.append("".join(current).strip())
        return sentences
    
    # Step 1: Split by double newline (paragraphs)
    paragraphs = text.split("\n\n")
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # Check if paragraph is a heading (single line, short)
        lines = paragraph.strip().split("\n")
        if len(lines) == 1 and len(lines[0]) < 50:  # Fixed: check length of first line, not the list
            # Treat as heading
            if current_size > 0:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = []
                current_size = 0
            heading_text = lines[0]  # Get the actual heading text
            current_chunk.append(heading_text)
            current_size += len(heading_text.split())
            continue
        
        # Process as paragraph
        words = paragraph.split()
        if current_size + len(words) <= max_chunk_size:
            current_chunk.extend(words)
            current_size += len(words)
        else:
            # Then we need to split by sentences
            sentences = split_by_sentences(paragraph)
            for sentence in sentences:
                sentence_words = sentence.split()
                if current_size + len(sentence_words) <= max_chunk_size:
                    current_chunk.extend(sentence_words)
                    current_size += len(sentence_words)
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk).strip())
                    current_chunk = sentence_words
                    current_size = len(sentence_words)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    
    return chunks

# Our example document
text = """Executive Summary

The company is evaluating multiple strategic initiatives for the coming fiscal year. 
Key priorities include market expansion, operational efficiency, and technology investment.

Strategic Initiatives

The chairman is considering investing in a new office in the GBA area. 
This initiative aligns with our broader regional growth strategy. 
The board has requested a detailed feasibility study within 90 days.

Operational Efficiency

We are also reviewing our supply chain operations to reduce costs. 
Automation and process optimization will be central to this effort."""

result = recursive_chunk(text, max_chunk_size=30)

for idx, chunk in enumerate(result, 1):
    print(f"{idx}. {chunk}\n")
```

#### After (Recursive Chunks):

```text
1. Executive Summary The company is evaluating multiple strategic initiatives for the coming fiscal year. Key priorities include market expansion, operational efficiency, and technology investment.
2. Strategic Initiatives The chairman is considering investing in a new office in the GBA area. This initiative aligns with our broader regional growth strategy.
3. The board has requested a detailed feasibility study within 90 days.
4. Operational Efficiency We are also reviewing our supply chain operations to reduce costs. Automation and process optimization will be central to this effort.
```

### Structure-Aware Chunking
Structure-aware chunking is very similar to recursive, but it take it one notch further by explicitly recognizing the 
structural elements of a document, such as headings, subheadings, tables, lists, figures, and code blocks. Rather than 
relying solely on text length or sentence boundaries, it uses the document's formatting and layout to determine chunk boundaries. 
In this approach, the primary objective is to preserve the meaning associated with each structural component so that retrieval remains 
aligned with the author's intended organization of information. A real-world example would be processing an annual 
report where the "Risk Factors" section, financial tables, and management discussion are each stored as separate chunks, 
even if their sizes differ significantly.

Let's have a look at a concrete example using structure-aware chunking on a simplified text:

#### Python Code: Structure-Aware Chunking

```python
def structure_aware_chunk(document):
    """
    Again here building a simple structure-aware chunker.

    Rules:
    - Headings start with '#'
    - Tables start with '|'
    - Lists start with '-'
    - Everything else belongs to the current section
    """

    chunks = []
    current_chunk = []

    for line in document.split("\n"):
        line = line.strip()

        if not line:
            continue

        # New heading starts a new chunk
        if line.startswith("#"):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            current_chunk.append(line)

        # Tables become independent chunks
        elif line.startswith("|"):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            chunks.append(line)

        # Lists become independent chunks
        elif line.startswith("-"):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            chunks.append(line)

        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


document = """
# Revenue Growth

Revenue increased by 15% during the fiscal year.
Growth was driven primarily by expansion in Asia.

| Region | Growth |
| Asia | 20% |
| Europe | 8% |

# Risk Factors

- Currency fluctuations
- Supply chain disruptions

Management continues to monitor these risks closely.
"""

chunks = structure_aware_chunk(document)

for idx, chunk in enumerate(chunks, 1):
    print(f"Chunk {idx}\n{chunk}\n")
```

#### After (Structure-Aware Chunks):

```text
Chunk 1
# Revenue Growth
Revenue increased by 15% during the fiscal year.
Growth was driven primarily by expansion in Asia.
Chunk 2
| Region | Growth |
Chunk 3
| Asia | 20% |
Chunk 4
| Europe | 8% |
Chunk 5
# Risk Factors
Chunk 6
- Currency fluctuations
Chunk 7
- Supply chain disruptions
Chunk 8
Management continues to monitor these risks closely.
```

Unlike fixed-size chunking, which would split the document every N tokens, or recursive chunking, which focuses on 
hierarchical text boundaries, structure-aware chunking explicitly preserves the semantic role of document elements. 
The structure-aware chunking approach would be particularly handy for annual reports, regulatory filings, research papers, and technical documentation, 
where tables, lists, and section headings often carry as much meaning as the surrounding narrative text - the famous footnotes :).

### Semantic Chunking

Semantic chunking represents a significant evolution from structure-aware approaches. Rather than relying solely on 
document formatting, headings, or predefined boundaries, it attempts to identify where the meaning of the text changes. 
The fundamental principle is that information discussing the same topic should remain together, while a shift in subject 
matter should trigger a new chunk. In this approach, the primary goal is to maximize semantic coherence within each 
chunk while trying to avoid as much as possible the mixing of unrelated concepts. In a real-world example, we can take again the case of annual report where several 
paragraphs discussing revenue growth remain grouped together, but a subsequent discussion about management changes or 
capital allocation begins a new chunk, even if no explicit section heading exists.

Semantic chunking is widerly used nowadays. Each paragraph or sentence is converted into a vector 
representation, and the similarity between consecutive vectors is measured. When similarity falls below a predefined 
threshold (that's where the trick lies), the system assumes that the topic has changed and starts a new chunk.

Let's take a look at a concrete example using semantic chunking with a simplified similarity calculation.

#### Python Code: Semantic Chunking (no external libraries)

```python
from collections import Counter
import math

def cosine_similarity(text1, text2):
    """
    For this example I will use a very simple cosine similarity using word frequencies.
    """
    
    words1 = Counter(text1.lower().split())
    words2 = Counter(text2.lower().split())

    vocabulary = set(words1.keys()).union(words2.keys())

    vec1 = [words1[word] for word in vocabulary]
    vec2 = [words2[word] for word in vocabulary]

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


def semantic_chunk(paragraphs, threshold=0.15):
    """
    Create a new chunk when similarity falls below threshold.
    """

    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        similarity = cosine_similarity(
            paragraphs[i - 1],
            paragraphs[i]
        )

        if similarity >= threshold:
            current_chunk.append(paragraphs[i])
        else:
            chunks.append("\n".join(current_chunk))
            current_chunk = [paragraphs[i]]

    chunks.append("\n".join(current_chunk))

    return chunks


paragraphs = [
    "Revenue increased by 15 percent during the year.",
    "Growth was strongest in the Asia Pacific region.",
    "Profit margins improved because of operating leverage.",
    "The board appointed a new chief executive officer.",
    "Management restructuring will continue next year.",
    "A share buyback program was approved by directors."
]

chunks = semantic_chunk(paragraphs)

for idx, chunk in enumerate(chunks, 1):
    print(f"Chunk {idx}\n{chunk}\n")
```

#### After (Semantic Chunks):

```text
Chunk 1
Revenue increased by 15 percent during the year.
Chunk 2
Growth was strongest in the Asia Pacific region.
Chunk 3
Profit margins improved because of operating leverage.
Chunk 4
The board appointed a new chief executive officer.
Chunk 5
Management restructuring will continue next year.
Chunk 6
A share buyback program was approved by directors.
```

Unlike recursive or structure-aware chunking, semantic chunking is not constrained by document formatting. Two paragraphs 
may belong to the same chunk even when separated by headings, while consecutive paragraphs may be split if they discuss 
different subjects. This makes semantic chunking particularly effective for earnings call transcripts, meeting notes, 
research reports, and other documents where topic transitions do not always align with explicit structural boundaries.

To understand what is happening behind the scenes, imagine that the algorithm calculates a similarity score between consecutive paragraphs.

Consider the following examples:

```text
Paragraph 1:
Revenue increased by 15 percent during the year.

Paragraph 2:
Growth was strongest in the Asia Pacific region.

Similarity Score: 0.87
```

Because both paragraphs discuss financial performance and growth, their similarity score is high. As a result, they remain in the same chunk.

Now consider:

```text
Paragraph 2:
Growth was strongest in the Asia Pacific region.

Paragraph 3:
The board appointed a new chief executive officer.

Similarity Score: 0.23
```

The topic has shifted from business performance to corporate governance. Since the similarity score falls below the configured threshold, a new chunk is created.

Conceptually, the algorithm behaves like this:

```python
similarity = compare(previous_paragraph, current_paragraph)

if similarity < threshold:
    start_new_chunk()
```

Using the above example, the similarity calculations might look like:

```text
Revenue increased by 15 percent during the year.
Growth was strongest in the Asia Pacific region.
Similarity = 0.87
OK Same Chunk

Growth was strongest in the Asia Pacific region.
Profit margins improved because of operating leverage.
Similarity = 0.81
OK Same Chunk

Profit margins improved because of operating leverage.
The board appointed a new chief executive officer.
Similarity = 0.23
WATCH: New Chunk

The board appointed a new chief executive officer.
Management restructuring will continue next year.
Similarity = 0.79
OK Same Chunk

Management restructuring will continue next year.
A share buyback program was approved by directors.
Similarity = 0.34
WATCH: New Chunk
```

So the objective of semantic chunking is therefore simple: keep information discussing the same topic together while 
separating information that belongs to different topics. Unlike fixed-size, recursive, or structure-aware chunking, 
the decision is driven by meaning rather than by token counts, headings, or document formatting.


### Parent-Child Chunking

Parent-child chunking introduces a two-tiered retrieval strategy that decouples the search process from the context that 
is ultimately fed to the generation model. The fundamental principle is that small, granular chunks (children) are used 
for efficient retrieval, but once relevant child chunks are identified, their larger parent chunks are returned to provide 
richer context. This approach addresses a core tension in chunking: smaller chunks improve retrieval precision by reducing 
noise, but larger chunks preserve the contextual information needed for coherent generation.

Let us formalize this relationship: child chunks (c_i) are generated by splitting the source material into fine-grained units, 
while parent chunks (p_j) are larger contextual containers that encompass one or more child chunks. The retrieval process 
operates on children, but the generation step receives the corresponding parents. The mapping can be expressed as:

retrieve(c_i) → return(p_j)

where p_j is the parent container of c_i.

A real-world example would be a corporate annual report where individual sentences (children) are indexed for search, 
but when a user asks about capital allocation, the system retrieves the relevant sentence and returns the entire Capital 
Allocation section (parent) to provide the necessary context for a comprehensive answer.

Let us examine a concrete example:

#### Python Code: Parent-Child Chunking

```python
def parent_child_chunk(text, child_size=3, parent_size=10):
    """
    Create parent-child chunking structure.
    Children are small chunks (sentences or token groups).
    Parents are larger chunks that contain multiple children.
    """
    # Split into sentences first (simplified)
    sentences = text.split('. ')
    sentences = [s + '.' for s in sentences if s.strip()]
    
    # Create child chunks (small, for retrieval)
    children = []
    current_child = []
    child_token_count = 0
    
    for sentence in sentences:
        tokens = len(sentence.split())
        if child_token_count + tokens <= child_size:
            current_child.append(sentence)
            child_token_count += tokens
        else:
            if current_child:
                children.append(' '.join(current_child))
            current_child = [sentence]
            child_token_count = tokens
    
    if current_child:
        children.append(' '.join(current_child))
    
    # Create parent chunks (large, for generation context)
    parents = []
    current_parent = []
    parent_token_count = 0
    
    for sentence in sentences:
        tokens = len(sentence.split())
        if parent_token_count + tokens <= parent_size:
            current_parent.append(sentence)
            parent_token_count += tokens
        else:
            if current_parent:
                parents.append(' '.join(current_parent))
            current_parent = [sentence]
            parent_token_count = tokens
    
    if current_parent:
        parents.append(' '.join(current_parent))
    
    # Map children to their parent chunks
    child_parent_map = {}
    for child in children:
        child_words = set(child.split())
        best_parent = None
        max_overlap = 0
        
        for idx, parent in enumerate(parents):
            parent_words = set(parent.split())
            overlap = len(child_words.intersection(parent_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_parent = idx
        
        if best_parent is not None:
            child_parent_map[child] = parents[best_parent]
    
    return children, parents, child_parent_map

# Example text about a company's financial performance
text = "Revenue increased by 15 percent during the fiscal year. Growth was strongest in the Asia Pacific region. Profit margins improved due to operational efficiency. The board appointed a new CEO in Q3. Management restructuring will continue through next year. The company also announced a share buyback program."

children, parents, mapping = parent_child_chunk(text, child_size=8, parent_size=25)

print("CHILDREN (Retrieval Index):")
for idx, child in enumerate(children, 1):
    print(f"  {idx}. {child}")

print("\nPARENTS (Generation Context):")
for idx, parent in enumerate(parents, 1):
    print(f"  {idx}. {parent}")

print("\nMAPPING (Child → Parent):")
for child, parent in list(mapping.items())[:2]:
    print(f"  Child: \"{child[:40]}...\"")
    print(f"    → Parent: \"{parent[:60]}...\"\n")
```

#### After (Parent-Child Structure):

```text
CHILDREN (Retrieval Index):
  1. Revenue increased by 15 percent during the fiscal year.
  2. Growth was strongest in the Asia Pacific region.
  3. Profit margins improved due to operational efficiency.
  4. The board appointed a new CEO in Q3.
  5. Management restructuring will continue through next year.
  6. The company also announced a share buyback program..
PARENTS (Generation Context):
  1. Revenue increased by 15 percent during the fiscal year. Growth was strongest in the Asia Pacific region. Profit margins improved due to operational efficiency.
  2. The board appointed a new CEO in Q3. Management restructuring will continue through next year. The company also announced a share buyback program..
MAPPING (Child → Parent):
  Child: "Revenue increased by 15 percent during t..."
    → Parent: "Revenue increased by 15 percent during the fiscal year. Grow..."
  Child: "Growth was strongest in the Asia Pacific..."
    → Parent: "Revenue increased by 15 percent during the fiscal year. Grow..."

```

Unlike fixed-size or recursive chunking, parent-child chunking does not simply segment text into uniform blocks. Instead, 
it creates a deliberate separation between what is searched (children) and what is presented to the model (parents). 
This dual structure provides the best of both worlds: the precision of fine-grained retrieval and the contextual richness 
required for faithful generation.

Parent-child chunking is particularly valuable for long-form documents such as legal contracts, financial reports, academic 
papers, and technical manuals, where a query often requires understanding not just a specific phrase but also the broader 
context in which that phrase appears.

### Graph-Based Chunking

I was a bit hesitant to discuss aboit Graph-nased chunking, just like Agentic Chunking, because there are actually relying on LLM-chucking.
Graph-based chunking represents a fundamental departure from previously mentioned linear segmentation methods. Rather than organizing text into 
sequential chunks, it models the source material as a knowledge graph where entities and their relationships become the 
fundamental units of information. The underlying principle is that meaning in text is often carried not by individual passages 
but by the connections between concepts, people, organizations, and events.

Formally, we can represent a document as a graph G = (V, E) where:

- V represents vertices, corresponding to entities extracted from the text (people, organizations, locations, concepts, etc.)
- E represents edges, corresponding to the relationships between these entities

There are, however, several limitations to graph-based chunking. The process of extracting entities and relationships 
typically relies on LLMs, making it significantly more computationally expensive than traditional chunking approaches. 
Its effectiveness is also highly dependent on the quality of entity extraction; missed entities or incorrectly inferred 
relationships can lead to incomplete or misleading graph structures. As the volume of data and the number of relationships increase, 
maintaining and querying the graph becomes increasingly complex. Finally, graph-based chunking is generally better suited 
to documents rich in explicit entities and relationships, such as technical documentation, corporate filings, or news articles, 
than to purely narrative or highly unstructured text.

A real-world example would be processing a news article about Microsoft's investment in OpenAI. Rather than chunking the 
article into paragraphs, graph-based chunking would extract the entities (Microsoft, OpenAI, Sam Altman, Satya Nadella) 
and their relationships ("invested in," "partnered with," "announced") to form a structured knowledge representation.

Let me go through a concrete example:

#### Python Code: Graph-Based Chunking

```python
import re
from collections import defaultdict

class GraphChunker:
    def __init__(self):
        self.entities = {}
        self.relationships = []
        self.entity_counter = 0
    
    def extract_entities(self, text):
        """Simple entity extraction using capitalization patterns and known terms."""
        # Simple entity detection: proper nouns and known entity patterns
        words = text.split()
        entities = []
        
        # Pattern for proper nouns (capitalized words)
        i = 0
        while i < len(words):
            if words[i][0].isupper() and len(words[i]) > 1:
                entity = words[i]
                i += 1
                # Combine multi-word entities
                while i < len(words) and words[i][0].isupper() and len(words[i]) > 1:
                    entity += " " + words[i]
                    i += 1
                if len(entity) > 1:  # Avoid single letters
                    entities.append(entity)
            else:
                i += 1
        
        # Add some known entities if not found
        known_entities = ["Microsoft", "OpenAI", "Google", "Apple", "Amazon"]
        for known in known_entities:
            if known in text and known not in entities:
                entities.append(known)
        
        return entities
    
    def extract_relationships(self, text, entities):
        """Extract simple relationships based on verb patterns between entities."""
        relationships = []
        
        # Common relationship patterns
        patterns = [
            (r'(\w+\s?\w*)\s+invested in\s+(\w+\s?\w*)', 'invested_in'),
            (r'(\w+\s?\w*)\s+acquired\s+(\w+\s?\w*)', 'acquired'),
            (r'(\w+\s?\w*)\s+partnered with\s+(\w+\s?\w*)', 'partnered_with'),
            (r'(\w+\s?\w*)\s+announced\s+(\w+\s?\w*)', 'announced'),
            (r'(\w+\s?\w*)\s+appointed\s+(\w+\s?\w*)', 'appointed')
        ]
        
        for pattern, rel_type in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Check if both parts are in our entity list or appear to be entities
                entity1 = match[0]
                entity2 = match[1]
                
                # Validate that both are entities
                if (entity1 in entities or entity1 in text) and (entity2 in entities or entity2 in text):
                    relationships.append({
                        'source': entity1,
                        'target': entity2,
                        'type': rel_type
                    })
        
        return relationships
    
    def build_graph(self, text):
        """Build a graph representation of the text."""
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text, entities)
        
        # Assign IDs to entities
        for entity in entities:
            if entity not in self.entities:
                self.entity_counter += 1
                self.entities[entity] = f"E{self.entity_counter}"
        
        return {
            'vertices': self.entities,
            'edges': relationships,
            'graph_summary': f"G = (V, E) where V={list(self.entities.values())} and E={len(relationships)} relationships"
        }

# Example text about a corporate investment
text = "Microsoft invested in OpenAI to accelerate AI development. Satya Nadella announced the partnership. The company also acquired several smaller startups."

chunker = GraphChunker()
graph = chunker.build_graph(text)

print("GRAPH REPRESENTATION:")
print(f"Vertices (Entities):")
for entity, entity_id in graph['vertices'].items():
    print(f"  {entity_id}: {entity}")

print(f"\nEdges (Relationships):")
for rel in graph['edges']:
    print(f"  {rel['source']} --{rel['type']}--> {rel['target']}")

print(f"\nGraph Summary:")
print(f"  {graph['graph_summary']}")
```

#### After (Graph Representation):

```text
GRAPH REPRESENTATION:
Vertices (Entities):
  E1: Microsoft
  E2: OpenAI
  E3: AI
  E4: Satya Nadella
  E5: The
Edges (Relationships):
  Microsoft --invested_in--> OpenAI to
  company also --acquired--> several smaller
  Satya Nadella --announced--> the partnership
Graph Summary:
  G = (V, E) where V=['E1', 'E2', 'E3', 'E4', 'E5'] and E=3 relationships
```

Unlike linear chunking methods that preserve the sequential order of text, graph-based chunking organizes information as a network of interconnected facts. This approach is exceptionally powerful for question answering that requires traversing relationships: "Which companies has Microsoft invested in?" can be answered by querying the graph's edges rather than searching through chunks of text. Graph-based chunking is particularly effective for processing news articles, corporate filings, research papers, and any document where relationships between entities carry more significance than the narrative flow.

### Hierarchical Chunking

Hierarchical chunking addresses the reality that documents naturally organize themselves at multiple levels of abstraction. Rather than flattening all text into a single granularity level, this method creates a multi-tiered structure that captures the document's inherent hierarchy. The fundamental principle is that information can be accessed at different levels of specificity depending on the query's needs.

We can formalize this hierarchy as a layered structure where each level represents progressively more granular information:

D → S → C → F

Where:
- D represents the Document level (entire document summary)
- S represents the Section level (major thematic divisions)
- C represents the Chunk level (paragraph-sized units)
- F represents the Fact level (individual statements or sentences)

A real-world example would be a company's annual report where the top level contains an executive summary, the next level contains major business segments, followed by paragraphs within each segment, and finally individual facts or data points. A query about "company performance" might retrieve from the Document level, while a query about "Q3 revenue in Asia" would navigate down to the Fact level.

As before, jumping into a small example:

#### Python Code: Hierarchical Chunking

```python
def hierarchical_chunk(document):
    """
    Create hierarchical chunks at multiple levels:
    Level 1: Document
    Level 2: Sections
    Level 3: Paragraphs
    Level 4: Sentences/Facts
    """
    
    # Split into sections (by double newline and headings)
    sections = []
    current_section = []
    current_heading = "Main Content"
    
    lines = document.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect headings (lines that are short and end without period)
        if len(line) < 50 and not line.endswith('.') and not line.endswith('?'):
            if current_section:
                sections.append({
                    'heading': current_heading,
                    'content': '\n'.join(current_section)
                })
            current_heading = line
            current_section = []
        else:
            current_section.append(line)
    
    if current_section:
        sections.append({
            'heading': current_heading,
            'content': '\n'.join(current_section)
        })
    
    # Build hierarchical structure
    hierarchy = {
        'document': document[:200] + '...' if len(document) > 200 else document,
        'sections': []
    }
    
    for section in sections:
        # Split section into paragraphs
        paragraphs = [p.strip() for p in section['content'].split('\n\n') if p.strip()]
        
        # Split paragraphs into sentences (facts)
        facts = []
        for para in paragraphs:
            sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
            facts.extend(sentences)
        
        hierarchy['sections'].append({
            'heading': section['heading'],
            'summary': section['content'][:100] + '...' if len(section['content']) > 100 else section['content'],
            'paragraphs': paragraphs,
            'facts': facts
        })
    
    return hierarchy

# Example document
document = """Executive Summary

The company achieved record revenue in fiscal year 2025. Overall performance exceeded market expectations.

Business Performance

Revenue growth was driven by international expansion. The Asia Pacific region contributed 35% of total revenue. Operating margins improved by 200 basis points.

Strategic Initiatives

The company announced a $5 billion share buyback program. Management also approved expansion into European markets. The board appointed a new Chief Technology Officer.

Risk Factors

Currency fluctuations remain a key concern. Supply chain disruptions could impact production. The company maintains a diversified supplier network."""

hierarchy = hierarchical_chunk(document)

print("HIERARCHICAL CHUNKING STRUCTURE:")
print("\nLevel 1: DOCUMENT")
print(f"  Document Summary: {hierarchy['document']}")

print("\nLevel 2: SECTIONS")
for idx, section in enumerate(hierarchy['sections'], 1):
    print(f"  Section {idx}: {section['heading']}")
    print(f"    Summary: {section['summary']}")
    
    print(f"\nLevel 3: PARAGRAPHS (within {section['heading']})")
    for p_idx, para in enumerate(section['paragraphs'], 1):
        print(f"    P{p_idx}: {para[:60]}...")
    
    print(f"\nLevel 4: FACTS (within {section['heading']})")
    for f_idx, fact in enumerate(section['facts'][:3], 1):  # Show first 3 facts
        print(f"    F{f_idx}: {fact}")
    if len(section['facts']) > 3:
        print(f"    ... and {len(section['facts']) - 3} more facts")
    print()

print("QUERY NAVIGATION EXAMPLES:")
print("  Query: 'Company performance 2025' → Navigate to Document Level")
print("  Query: 'Asia Pacific revenue' → Navigate to Section Level (Business Performance)")
print("  Query: 'Operating margins' → Navigate to Fact Level")
```

#### After (Hierarchical Structure):

```text
HIERARCHICAL CHUNKING STRUCTURE:
Level 1: DOCUMENT
  Document Summary: Executive Summary
The company achieved record revenue in fiscal year 2025. Overall performance exceeded market expectations.
Business Performance
Revenue growth was driven by international expansio...
Level 2: SECTIONS
  Section 1: Executive Summary
    Summary: The company achieved record revenue in fiscal year 2025. Overall performance exceeded market expecta...
Level 3: PARAGRAPHS (within Executive Summary)
    P1: The company achieved record revenue in fiscal year 2025. Ove...
Level 4: FACTS (within Executive Summary)
    F1: The company achieved record revenue in fiscal year 2025.
    F2: Overall performance exceeded market expectations.
  Section 2: Business Performance
    Summary: Revenue growth was driven by international expansion. The Asia Pacific region contributed 35% of tot...
Level 3: PARAGRAPHS (within Business Performance)
    P1: Revenue growth was driven by international expansion. The As...
Level 4: FACTS (within Business Performance)
    F1: Revenue growth was driven by international expansion.
    F2: The Asia Pacific region contributed 35% of total revenue.
    F3: Operating margins improved by 200 basis points.
  Section 3: Strategic Initiatives
    Summary: The company announced a $5 billion share buyback program. Management also approved expansion into Eu...
Level 3: PARAGRAPHS (within Strategic Initiatives)
    P1: The company announced a $5 billion share buyback program. Ma...
Level 4: FACTS (within Strategic Initiatives)
    F1: The company announced a $5 billion share buyback program.
    F2: Management also approved expansion into European markets.
    F3: The board appointed a new Chief Technology Officer.
  Section 4: Risk Factors
    Summary: Currency fluctuations remain a key concern. Supply chain disruptions could impact production. The co...
Level 3: PARAGRAPHS (within Risk Factors)
    P1: Currency fluctuations remain a key concern. Supply chain dis...
Level 4: FACTS (within Risk Factors)
    F1: Currency fluctuations remain a key concern.
    F2: Supply chain disruptions could impact production.
    F3: The company maintains a diversified supplier network.
QUERY NAVIGATION EXAMPLES:
  Query: 'Company performance 2025' → Navigate to Document Level
  Query: 'Asia Pacific revenue' → Navigate to Section Level (Business Performance)
  Query: 'Operating margins' → Navigate to Fact Level
```

Unlike semantic chunking, which focuses on a single optimal segmentation, hierarchical chunking preserves all levels of granularity simultaneously. Unlike structure-aware chunking, it doesn't just recognize document formatting—it explicitly organizes the document into a navigable tree structure. This multi-level approach is particularly powerful for applications like document question answering where queries can range from broad ("summarize the document") to highly specific ("What was the operating margin improvement in basis points?"), allowing the system to retrieve information at the most appropriate level of abstraction.

Hierarchical chunking is especially valuable for processing corporate reports, regulatory filings, technical specifications, and any document where information is naturally organized at multiple levels of detail and where users may have questions at any of these levels.

## Choosing the Right Chunking Strategy

With the variety of chunking methods now available, the question is not always which one is best, but which one is best for a given use case. In practice, RAG systems often benefit from combining methods rather than relying on a single strategy end to end. A fixed-size approach may be useful at one stage, while a structure-aware or semantic approach may be more appropriate at another, depending on the document type, the retrieval objective, and the expected query patterns. The table below is therefore meant as a practical guide to the relative strengths of each method, rather than as a rigid decision framework.

| Method | Best For | Trade-offs |
|--------|----------|------------|
| Fixed-Size | Simple documents, prototyping | Fastest, but can break context and fragment meaning |
| Recursive | Well-structured documents with clear hierarchy | Preserves structure, but depends on consistent formatting |
| Structure-Aware | Documents with tables, lists, headings, and other formatting | Preserves document elements, but may create uneven chunk sizes |
| Semantic | Unstructured documents with topic shifts | Groups related content more intelligently, but is more complex and model-dependent |
| Parent-Child | Queries that need both precision and context | Balances retrieval accuracy and context, but increases storage and implementation complexity |
| Graph-Based | Documents centered on entities and relationships | Strong for relationship queries, but difficult to implement and maintain |
| Hierarchical | Multi-level documents with varying query granularity | Flexible across query types, but more expensive to process |

In practice, these methods are often combined. For example, a document may first be split using structure-aware chunking to preserve headings and sections, then refined with fixed-size or semantic chunking inside each section. This allows the system to retain the document’s natural organization while still producing chunks that are small enough for efficient retrieval and generation.

A simple example would be an annual report. The report could first be divided into sections such as *Business Overview*, *Risk Factors*, and *Financial Statements* using structure-aware chunking. Within each section, fixed-size or semantic chunking could then be applied to break down long paragraphs into smaller units that are easier to retrieve. In this setup, the first method preserves the document structure, while the second improves search precision and retrieval efficiency.

## Conclusion

Chunking, often dismissed as a mundane preprocessing step, deserves far more attention than it typically receives in discussions about RAG systems. As I have attempted to demonstrate, the choice of chunking strategy—from simple fixed-size splitting to sophisticated graph-based and hierarchical approaches—profoundly influences the accuracy, cost, context preservation, and efficiency of the entire pipeline.

What makes chunking particularly challenging is that there is no universal optimal strategy. The terrain of your source material, the load your system must bear, and the judgment of the engineer will determine the most appropriate approach. A chunking strategy that works brilliantly for legal document retrieval may fail catastrophically for social media analysis. The key insight is to treat chunking not as a technical formality but as a design decision that requires careful consideration and, ideally, empirical validation against your specific use case.

In the next article of this series, we will explore how these chunking strategies interact with the retrieval component of the RAG pipeline, examining how different indexing and search algorithms can be tuned to work optimally with your chosen chunking approach. As we will see, the interface between chunking and retrieval is where much of the magic—and many of the challenges—of building production-grade RAG systems truly lies.