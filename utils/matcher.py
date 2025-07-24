def match_entities_and_update_relationships(
                                            self,
                                            entities1: List[Entity],
                                            entities2: List[Entity],
                                            relationships1: List[Relationship],
                                            relationships2: List[Relationship],
                                            rel_threshold: float = 0.8,
                                            ent_threshold: float = 0.8
                                        ) -> Tuple[List[Entity], List[Relationship]]:
    """
    Match two lists of entities (Entities) and update the relationships list accordingly.
    :param entities1: First list of entities to match.
    :param entities2: Second list of entities to match against.
    :param relationships1: First list of relationships to update.
    :param relationships2: Second list of relationships to compare.
    :param rel_threshold: Cosine similarity threshold for relationships.
    :param ent_threshold: Cosine similarity threshold for entities.
    :return: Updated entities list and relationships list.
    """
    # Step 1: Match the entities and relations from both lists
    matched_entities1, global_entities = self.process_lists(entities1, entities2, ent_threshold)
    matched_relations, _ = self.process_lists(relationships1, relationships2, rel_threshold)

    # Step 2: 使用字典存储实体，确保唯一性
    unique_entities = {}
    for entity in global_entities:
        # 使用实体名称和标签的组合作为唯一标识
        entity_key = (entity.name, entity.label)
        if entity_key not in unique_entities:
            unique_entities[entity_key] = entity
        else:
            # 如果存在相同的实体，保留嵌入向量相似度更高的那个
            existing_entity = unique_entities[entity_key]
            if cosine_similarity(
                np.array(entity.properties.embeddings).reshape(1, -1),
                np.array(existing_entity.properties.embeddings).reshape(1, -1)
            )[0][0] > ent_threshold:
                unique_entities[entity_key] = entity

    # 更新全局实体列表
    global_entities = list(unique_entities.values())

    # Step 3: Create a mapping from old entity names to matched entity names
    entity_name_mapping = {
        entity: matched_entity 
        for entity, matched_entity in zip(entities1, matched_entities1) 
        if entity != matched_entity
    }

    # Step 4: Update relationships based on matched entities
    def update_relationships(relationships: List[Relationship]) -> List[Relationship]:
        updated_relationships = []
        seen_relationships = set()  # 用于跟踪已处理的关系
        
        for rel in relationships:
            updated_rel = rel.model_copy()  # Create a copy to modify
            # Update the 'startEntity' and 'endEntity' names with matched entity names
            if rel.startEntity in entity_name_mapping:
                updated_rel.startEntity = entity_name_mapping[rel.startEntity]
            if rel.endEntity in entity_name_mapping:
                updated_rel.endEntity = entity_name_mapping[rel.endEntity]
            
            # 创建关系的唯一标识
            rel_key = (updated_rel.name, updated_rel.startEntity.name, updated_rel.endEntity.name)
            if rel_key not in seen_relationships:
                updated_relationships.append(updated_rel)
                seen_relationships.add(rel_key)
        
        return updated_relationships

    # Step 5: 更新并去重关系
    updated_relations = update_relationships(matched_relations)
    final_relationships = []
    seen_relationships = set()
    
    # 合并并去重所有关系
    for rel in relationships2 + updated_relations:
        rel_key = (rel.name, rel.startEntity.name, rel.endEntity.name)
        if rel_key not in seen_relationships:
            final_relationships.append(rel)
            seen_relationships.add(rel_key)

    return global_entities, final_relationships