"""RAG Pipeline - Syncs data from PostgreSQL to Pinecone.

Handles incremental updates with no duplicates or missing data.
"""

import logging
from typing import Dict, List, Any, Optional, Set

from database.postgres_client import PostgresClient
from vector_store.pinecone_client import PineconeClient
from rag.embedding_service import EmbeddingService
from rag.chunking_service import ChunkingService
from rag.entity_extractors import EXTRACTORS, EntityExtractor

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG Pipeline for syncing PostgreSQL data to Pinecone."""
    
    def __init__(
        self,
        db_client: PostgresClient,
        pinecone_client: PineconeClient,
        embedding_service: EmbeddingService,
        chunking_service: ChunkingService,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimension: int = 1024,
        batch_size: int = 100
    ):
        """Initialize RAG pipeline.
        
        Args:
            db_client: PostgreSQL client
            pinecone_client: Pinecone client
            embedding_service: Embedding service
            chunking_service: Chunking service
            embedding_model: Embedding model name
            embedding_dimension: Embedding dimension
        """
        self.db = db_client
        self.pinecone = pinecone_client
        self.embedding_service = embedding_service
        self.chunking_service = chunking_service
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
    
    def get_embedded_entities(
        self,
        entity_type: str,
        embedding_model: Optional[str] = None
    ) -> Set[str]:
        """Get set of entity IDs that already have embeddings.
        
        Args:
            entity_type: Type of entity (e.g., 'aircraft_listing', 'document')
            embedding_model: Embedding model name (defaults to instance model)
            
        Returns:
            Set of entity IDs (as strings) that are already embedded
        """
        model = embedding_model or self.embedding_model
        
        query = """
            SELECT DISTINCT entity_id
            FROM embeddings_metadata
            WHERE entity_type = %s 
              AND embedding_model = %s
              AND entity_id IS NOT NULL
        """
        
        results = self.db.execute_query(query, (entity_type, model))
        return {str(row['entity_id']) for row in results if row.get('entity_id')}
    
    def get_embedded_documents(
        self,
        embedding_model: Optional[str] = None
    ) -> Set[str]:
        """Get set of document IDs that already have embeddings.
        
        Args:
            embedding_model: Embedding model name (defaults to instance model)
            
        Returns:
            Set of document IDs (as strings) that are already embedded
        """
        model = embedding_model or self.embedding_model
        
        query = """
            SELECT DISTINCT document_id
            FROM embeddings_metadata
            WHERE document_id IS NOT NULL
              AND embedding_model = %s
        """
        
        results = self.db.execute_query(query, (model,))
        return {str(row['document_id']) for row in results if row.get('document_id')}
    
    def process_entity_type(
        self,
        entity_type: str,
        limit: Optional[int] = None,
        force_reembed: bool = False
    ) -> Dict[str, Any]:
        """Process a specific entity type.
        
        Args:
            entity_type: Type of entity to process
            limit: Optional limit on number of records
            force_reembed: If True, re-embed even if already embedded
            
        Returns:
            Statistics dictionary
        """
        if entity_type not in EXTRACTORS:
            logger.error(f"Unknown entity type: {entity_type}")
            return {'error': f"Unknown entity type: {entity_type}"}
        
        extractor = EXTRACTORS[entity_type]
        stats = {
            'entity_type': entity_type,
            'processed': 0,
            'embedded': 0,
            'skipped': 0,
            'errors': 0,
            'chunks_created': 0,
            'vectors_upserted': 0
        }
        
        # Get already embedded entities
        embedded_ids = set()
        if not force_reembed:
            embedded_ids = self.get_embedded_entities(entity_type)
            logger.info(f"Found {len(embedded_ids)} already embedded {entity_type} records")
        
        # Fetch records from database
        records = self._fetch_entity_records(entity_type, limit, embedded_ids)
        logger.info(f"Fetched {len(records)} {entity_type} records to process")
        
        # Process each record
        vectors_to_upsert = []
        metadata_to_insert = []
        
        for record in records:
            try:
                stats['processed'] += 1
                entity_id = str(record.get('id'))
                
                # Skip if already embedded (unless force_reembed)
                if entity_id in embedded_ids and not force_reembed:
                    stats['skipped'] += 1
                    continue
                
                # Extract text
                text = extractor.extract_text(record)
                if not text:
                    logger.debug(f"No extractable text for {entity_type} {entity_id}")
                    stats['skipped'] += 1
                    continue
                
                # Skip extremely large texts to avoid memory issues
                MAX_TEXT_LENGTH = 1_000_000  # 1MB of text
                if len(text) > MAX_TEXT_LENGTH:
                    logger.warning(f"Text too large for {entity_type} {entity_id} ({len(text)} chars), skipping")
                    stats['skipped'] += 1
                    continue
                
                # Chunk text if needed
                base_metadata = extractor.get_metadata(record)
                try:
                    chunks = self.chunking_service.chunk_text(text, base_metadata, chunk_id_prefix=f"{entity_type}_{entity_id}")
                except MemoryError:
                    logger.warning(f"Memory error chunking {entity_type} {entity_id} (text length: {len(text)}), skipping")
                    stats['skipped'] += 1
                    continue
                
                if not chunks:
                    stats['skipped'] += 1
                    continue
                
                stats['chunks_created'] += len(chunks)
                
                # Generate embeddings for chunks
                chunk_texts = [chunk['text'] for chunk in chunks]
                embeddings = self.embedding_service.embed_batch(chunk_texts)
                
                # Create vectors for Pinecone
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding is None:
                        continue
                    
                    # Create unique vector ID
                    vector_id = f"{entity_type}_{entity_id}_chunk_{chunk['chunk_index']}"
                    
                    # Filter out None/null values from metadata (Pinecone doesn't accept null)
                    clean_metadata = {
                        k: v for k, v in chunk['metadata'].items() 
                        if v is not None and v != '' and not (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')))
                    }
                    
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': clean_metadata
                    })
                
                # Track metadata for database
                metadata_to_insert.append({
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'document_id': None,  # Only for documents
                    'embedding_model': self.embedding_model,
                    'embedding_dimension': self.embedding_dimension,
                    'chunk_count': len(chunks),
                    'vector_store': 'pinecone',
                    'vector_store_id': f"{entity_type}_{entity_id}"
                })
                
                stats['embedded'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {entity_type} record {record.get('id')}: {e}", exc_info=True)
                stats['errors'] += 1
        
        # Upsert vectors to Pinecone in batches
        if vectors_to_upsert:
            try:
                upserted = self.pinecone.upsert_vectors(
                    vectors_to_upsert,
                    batch_size=self.batch_size
                )
                stats['vectors_upserted'] = upserted
            except Exception as e:
                logger.error(f"Failed to upsert vectors to Pinecone: {e}", exc_info=True)
                stats['errors'] += len(vectors_to_upsert)
        
        # Insert metadata into database
        if metadata_to_insert:
            self._insert_embedding_metadata(metadata_to_insert)
        
        return stats
    
    def _fetch_entity_records(
        self,
        entity_type: str,
        limit: Optional[int] = None,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch records from database for entity type.
        
        Args:
            entity_type: Type of entity
            limit: Optional limit
            exclude_ids: Set of IDs to exclude (already embedded)
            
        Returns:
            List of records
        """
        exclude_ids = exclude_ids or set()
        
        if entity_type == 'aircraft_listing':
            query = """
                SELECT * FROM aircraft_listings
                ORDER BY updated_at DESC, created_at DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            return self.db.execute_query(query)
        
        elif entity_type == 'document':
            query = """
                SELECT * FROM documents
                WHERE extracted_text IS NOT NULL AND extracted_text != ''
                ORDER BY updated_at DESC, created_at DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            return self.db.execute_query(query)
        
        elif entity_type == 'aircraft':
            query = """
                SELECT * FROM aircraft
                ORDER BY updated_at DESC, created_at DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            return self.db.execute_query(query)
        
        elif entity_type == 'aircraft_sale':
            query = """
                SELECT * FROM aircraft_sales
                ORDER BY created_at DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            return self.db.execute_query(query)
        
        elif entity_type == 'faa_registration':
            query = """
                SELECT * FROM faa_registrations
                ORDER BY updated_at DESC, created_at DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            return self.db.execute_query(query)
        
        else:
            logger.warning(f"No query defined for entity type: {entity_type}")
            return []
    
    def _insert_embedding_metadata(self, metadata_list: List[Dict[str, Any]]) -> None:
        """Insert embedding metadata into database.
        
        Args:
            metadata_list: List of metadata dictionaries
        """
        for metadata in metadata_list:
            try:
                # Check if metadata already exists
                check_query = """
                    SELECT id FROM embeddings_metadata
                    WHERE entity_type = %s 
                      AND entity_id = %s 
                      AND embedding_model = %s
                    LIMIT 1
                """
                exists = self.db.execute_query(
                    check_query,
                    (metadata['entity_type'], metadata['entity_id'], metadata['embedding_model'])
                )
                
                if exists:
                    # Update existing
                    update_query = """
                        UPDATE embeddings_metadata SET
                            chunk_count = %s,
                            vector_store_id = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE entity_type = %s 
                          AND entity_id = %s 
                          AND embedding_model = %s
                    """
                    self.db.execute_update(
                        update_query,
                        (
                            metadata['chunk_count'],
                            metadata['vector_store_id'],
                            metadata['entity_type'],
                            metadata['entity_id'],
                            metadata['embedding_model']
                        )
                    )
                else:
                    # Insert new
                    insert_query = """
                        INSERT INTO embeddings_metadata (
                            entity_type, entity_id, document_id,
                            embedding_model, embedding_dimension,
                            chunk_count, vector_store, vector_store_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    self.db.execute_update(
                        insert_query,
                        (
                            metadata['entity_type'],
                            metadata['entity_id'],
                            metadata.get('document_id'),
                            metadata['embedding_model'],
                            metadata['embedding_dimension'],
                            metadata['chunk_count'],
                            metadata['vector_store'],
                            metadata['vector_store_id']
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to insert embedding metadata: {e}", exc_info=True)
    
    def sync_all(
        self,
        entity_types: Optional[List[str]] = None,
        limit: Optional[int] = None,
        force_reembed: bool = False
    ) -> Dict[str, Any]:
        """Sync all entity types to Pinecone.
        
        Args:
            entity_types: List of entity types to sync (None = all)
            limit: Optional limit per entity type
            force_reembed: If True, re-embed even if already embedded
            
        Returns:
            Summary statistics
        """
        if entity_types is None:
            entity_types = list(EXTRACTORS.keys())
        
        summary = {
            'total_processed': 0,
            'total_embedded': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'total_chunks': 0,
            'total_vectors': 0,
            'entity_stats': {}
        }
        
        for entity_type in entity_types:
            logger.info(f"Processing entity type: {entity_type}")
            stats = self.process_entity_type(entity_type, limit, force_reembed)
            summary['entity_stats'][entity_type] = stats
            summary['total_processed'] += stats.get('processed', 0)
            summary['total_embedded'] += stats.get('embedded', 0)
            summary['total_skipped'] += stats.get('skipped', 0)
            summary['total_errors'] += stats.get('errors', 0)
            summary['total_chunks'] += stats.get('chunks_created', 0)
            summary['total_vectors'] += stats.get('vectors_upserted', 0)
        
        return summary
