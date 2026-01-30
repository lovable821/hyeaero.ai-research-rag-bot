"""Pinecone vector database client for RAG pipeline."""

import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from pinecone.exceptions import PineconeException

logger = logging.getLogger(__name__)


class PineconeClient:
    """Pinecone vector database client."""
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1024,
        metric: str = "cosine",
        host: Optional[str] = None,
    ):
        """Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Vector dimension (default: 1024 for text-embedding-3-large)
            metric: Distance metric (default: cosine)
            host: Optional custom host URL
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.host = host
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index = None
        
    def connect(self) -> bool:
        """Connect to Pinecone index.
        
        Returns:
            True if connection successful
        """
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.error(f"Pinecone index '{self.index_name}' does not exist")
                return False
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            return True
        except PineconeException as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            return False
    
    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> int:
        """Upsert vectors to Pinecone.
        
        Args:
            vectors: List of vectors with format:
                {
                    'id': str,
                    'values': List[float],
                    'metadata': Dict[str, Any]
                }
            namespace: Optional namespace
            batch_size: Batch size for upserts
            
        Returns:
            Number of vectors upserted
        """
        if not self.index:
            raise RuntimeError("Not connected to Pinecone index. Call connect() first.")
        
        total_upserted = 0
        
        # Upsert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
            except PineconeException as e:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Successfully upserted {total_upserted} vectors to Pinecone")
        return total_upserted
    
    def delete_vectors(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> bool:
        """Delete vectors from Pinecone.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Optional namespace
            
        Returns:
            True if successful
        """
        if not self.index:
            raise RuntimeError("Not connected to Pinecone index. Call connect() first.")
        
        try:
            self.index.delete(ids=ids, namespace=namespace)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
            return True
        except PineconeException as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query Pinecone for similar vectors.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            namespace: Optional namespace
            filter: Optional metadata filter
            
        Returns:
            List of similar vectors with scores
        """
        if not self.index:
            raise RuntimeError("Not connected to Pinecone index. Call connect() first.")
        
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True
            )
            return results.matches
        except PineconeException as e:
            logger.error(f"Failed to query Pinecone: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            raise RuntimeError("Not connected to Pinecone index. Call connect() first.")
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': dict(stats.namespaces) if stats.namespaces else {}
            }
        except PineconeException as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {}
