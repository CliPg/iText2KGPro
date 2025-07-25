from typing import List
from .ientities_extraction import iEntitiesExtractor
from .irelations_extraction import iRelationsExtractor
from .utils import Matcher, LangchainOutputParser
from .models import KnowledgeGraph
import json
import numpy as np
import os

class iText2KG:
    """
    A class designed to extract knowledge from text and structure it into a knowledge graph using
    entity and relationship extraction powered by language models.
    """
    def __init__(self, llm_model, embeddings_model, sleep_time:int=5) -> None:        
        """
        Initializes the iText2KG with specified language model, embeddings model, and operational parameters.
        
        Args:
        llm_model: The language model instance to be used for extracting entities and relationships from text.
        embeddings_model: The embeddings model instance to be used for creating vector representations of extracted entities.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """
        self.ientities_extractor =  iEntitiesExtractor(llm_model=llm_model, 
                                                       embeddings_model=embeddings_model,
                                                       sleep_time=sleep_time) 
        
        self.irelations_extractor = iRelationsExtractor(llm_model=llm_model, 
                                                        embeddings_model=embeddings_model,
                                                        sleep_time=sleep_time)

        self.matcher = Matcher()
        self.langchain_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=embeddings_model)


    def build_graph(self, 
                    sections:List[str], 
                    existing_knowledge_graph:KnowledgeGraph=None, 
                    ent_threshold:float = 0.7, 
                    rel_threshold:float = 0.7, 
                    max_tries:int=5, 
                    max_tries_isolated_entities:int=3,
                    entity_name_weight:float=0.6,
                    entity_label_weight:float=0.4,
                    ) -> KnowledgeGraph:
        """
        Builds a knowledge graph from text by extracting entities and relationships, then integrating them into a structured graph.
        This function leverages language models to extract and merge knowledge from multiple sections of text.

        Args:
        sections (List[str]): A list of strings where each string represents a section of the document from which entities 
                              and relationships will be extracted.
        existing_knowledge_graph (KnowledgeGraph, optional): An existing knowledge graph to merge the newly extracted 
                                                             entities and relationships into. Default is None.
        ent_threshold (float, optional): The threshold for entity matching, used to merge entities from different 
                                         sections. A higher value indicates stricter matching. Default is 0.7.
        rel_threshold (float, optional): The threshold for relationship matching, used to merge relationships from 
                                         different sections. Default is 0.7.
        entity_name_weight (float): The weight of the entity name, set to 0.6, indicating its
                                     relative importance in the overall evaluation process.
        entity_label_weight (float): The weight of the entity label, set to 0.4, reflecting its
                                      secondary significance in the evaluation process.
        max_tries (int, optional): The maximum number of attempts to extract entities and relationships. Defaults to 5.
        max_tries_isolated_entities (int, optional): The maximum number of attempts to process isolated entities 
                                                     (entities without relationships). Defaults to 3.
        

        Returns:
        KnowledgeGraph: A constructed knowledge graph consisting of the merged entities and relationships extracted 
                        from the text.
        """

        """
        用sections[0]初始化global entities and relationships
        """
        print("[INFO] ------- Extracting Entities from the Document", 1)
        global_entities = self.ientities_extractor.extract_entities(context=sections[0],
                                                                    entity_name_weight= entity_name_weight,
                                                                    entity_label_weight=entity_label_weight)
        print("[INFO] ------- Extracting Relations from the Document", 1)
        global_relationships = self.irelations_extractor.extract_verify_and_correct_relations(context=sections[0], 
                                                                                              entities = global_entities, 
                                                                                              rel_threshold=rel_threshold, 
                                                                                              max_tries=max_tries, 
                                                                                              max_tries_isolated_entities=max_tries_isolated_entities,
                                                                                              entity_name_weight= entity_name_weight,
                                                                                              entity_label_weight=entity_label_weight)
        
                
        for i in range(1, len(sections)):
            print("[INFO] ------- Extracting Entities from the Document", i+1)
            entities = self.ientities_extractor.extract_entities(context= sections[i],
                                                                 entity_name_weight= entity_name_weight,
                                                                 entity_label_weight=entity_label_weight)
            if not entities:
                continue
            processed_entities, global_entities = self.matcher.process_lists(list1 = entities, list2=global_entities, threshold=ent_threshold)
            
            """
            # 去除global_entities中的重复实体
            unique_entities = []
            seen_names = set()
            
            for entity in global_entities:
                if entity.name not in seen_names:
                    unique_entities.append(entity)
                    seen_names.add(entity.name)
            
            global_entities = unique_entities
            """

            print("[INFO] ------- Extracting Relations from the Document", i+1)
            relationships = self.irelations_extractor.extract_verify_and_correct_relations(context= sections[i], 
                                                                                           entities=processed_entities, 
                                                                                           rel_threshold=rel_threshold,
                                                                                           max_tries=max_tries, 
                                                                                           max_tries_isolated_entities=max_tries_isolated_entities,
                                                                                           entity_name_weight= entity_name_weight,
                                                                                           entity_label_weight=entity_label_weight)
            processed_relationships, _ = self.matcher.process_lists(list1 = relationships, list2=global_relationships, threshold=rel_threshold)
            
            global_relationships.extend(processed_relationships)
        
        if existing_knowledge_graph:                    
            print(f"[INFO] ------- Matching the Document {1} Entities and Relationships with the Existing Global Entities/Relations")
            global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(entities1=global_entities,
                                                                 entities2=existing_knowledge_graph.entities,
                                                                 relationships1=global_relationships,
                                                                 relationships2=existing_knowledge_graph.relationships,
                                                                 ent_threshold=ent_threshold,
                                                                 rel_threshold=rel_threshold)

            """
            unique_entities = []
            seen_names = set()
            
            for entity in global_entities:
                if entity.name not in seen_names:
                    unique_entities.append(entity)
                    seen_names.add(entity.name)
            
            global_entities = unique_entities
            """   
            
        constructed_kg = KnowledgeGraph(entities=global_entities, relationships=global_relationships)
        constructed_kg.remove_duplicates_entities()
        constructed_kg.remove_duplicates_relationships()
         
        return constructed_kg
    

    def convert_ndarray_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_ndarray_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarray_to_list(item) for item in obj]
        else:
            return obj
    

    def save_graph(self, kg: KnowledgeGraph, filename: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        kgs_dir = os.path.join(os.path.dirname(current_dir), "kgs")
        os.makedirs(kgs_dir, exist_ok=True)

        base_filename = os.path.basename(filename)
        save_path = os.path.join(kgs_dir, base_filename)

        raw_data = kg.model_dump()
        safe_data = self.convert_ndarray_to_list(raw_data)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(safe_data, f, ensure_ascii=False, indent=2)


    
    def load_graph(self, filename: str) -> KnowledgeGraph:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        kgs_dir = os.path.join(project_root, "kgs")
        load_path = os.path.join(kgs_dir, filename)
    
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        return KnowledgeGraph.model_validate(data)

    