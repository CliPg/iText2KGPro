from itext2kg import iText2KG
from langchain.document_loaders import PyPDFLoader
from itext2kg.documents_distiller import DocumentsDistiller, Article
from pydantic import BaseModel, Field
from typing import List, Tuple
from itext2kg.graph_integration import GraphIntegrator
from langchain_ollama import ChatOllama, OllamaEmbeddings


llm = ChatOllama(
    model="llama3",
    temperature=0,
)

embeddings = OllamaEmbeddings(
    model="llama3",
)

class ArticleResults(BaseModel):
    abstract:str = Field(description="Brief summary of the article's abstract")
    key_findings:str = Field(description="The key findings of the article")
    limitation_of_sota : str=Field(description="limitation of the existing work")
    proposed_solution : str = Field(description="the proposed solution in details")
    paper_limitations : str=Field(description="The limitations of the proposed solution of the paper")

# Sample input data as a list of triplets
# It is structured in this manner : (document's path, page_numbers_to_exclude, blueprint, document_type)
documents_information = [
    ("../datasets/llm-tikg.pdf", [11,10], ArticleResults, 'scientific article'),
    ("../datasets/actionable-cyber-threat.pdf", [12,11,10], ArticleResults, 'scientific article')
]


def upload_and_distill(documents_information: List[Tuple[str, List[int], BaseModel]]):
    distilled_docs = []
    
    for path_, exclude_pages, blueprint, document_type in documents_information:
        
        loader = PyPDFLoader(path_)
        pages = loader.load_and_split()
        pages = [page for page in pages if page.metadata["page"]+1 not in exclude_pages] # Exclude some pages (unecessary pages, for example, the references)
        document_distiller = DocumentsDistiller(llm_model=llm)
        
        IE_query = f'''
        # DIRECTIVES : 
        - Act like an experienced information extractor.
        - You have a chunk of a {document_type}
        - If you do not find the right information, keep its place empty.
        '''
        
        # Distill document content with query
        distilled_doc = document_distiller.distill(
            documents=[page.page_content.replace("{", '[').replace("}", "]") for page in pages],
            IE_query=IE_query,
            output_data_structure=blueprint
        )
        
        # Filter and format distilled document results
        distilled_docs.append([
            f"{document_type}'s {key} - {value}".replace("{", "[").replace("}", "]") 
            for key, value in distilled_doc.items() 
            if value and value != []
        ])
    
    return distilled_docs

distilled_docs = upload_and_distill(documents_information=documents_information)

itext2kg = iText2KG(llm_model = llm, embeddings_model = embeddings)
kg = itext2kg.build_graph(sections=distilled_docs[0], ent_threshold=0.7, rel_threshold=0.7)
kg2 = itext2kg.build_graph(sections=distilled_docs[1], existing_knowledge_graph=kg, rel_threshold=0.7, ent_threshold=0.7)



URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "LHSqaz123"

GraphIntegrator(uri=URI, username=USERNAME, password=PASSWORD).visualize_graph(knowledge_graph=kg2)

