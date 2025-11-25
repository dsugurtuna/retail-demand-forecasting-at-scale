"""
Feature Store for managing computed features.

Provides centralized feature storage and retrieval with:
- Versioning support
- Point-in-time correctness
- Feature lineage tracking
- Caching layer
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FeatureMetadata(BaseModel):
    """Metadata for a feature set."""
    
    name: str
    version: str
    created_at: datetime
    features: list[str]
    row_count: int
    source_hash: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class FeatureStore:
    """
    Feature Store for managing computed features.
    
    Provides:
    - Centralized feature storage and retrieval
    - Version control for feature sets
    - Point-in-time feature retrieval
    - Caching for performance
    - Feature lineage tracking
    
    Example:
        >>> store = FeatureStore("data/features")
        >>> store.save("training_features", features, version="v1.0")
        >>> loaded = store.load("training_features", version="v1.0")
    """
    
    def __init__(
        self,
        storage_path: str | Path,
        cache_enabled: bool = True
    ) -> None:
        """
        Initialize Feature Store.
        
        Args:
            storage_path: Path to feature storage directory
            cache_enabled: Whether to enable in-memory caching
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.cache_enabled = cache_enabled
        self._cache: dict[str, pl.DataFrame] = {}
        self._metadata_cache: dict[str, FeatureMetadata] = {}
    
    def save(
        self,
        name: str,
        features: pl.DataFrame,
        version: str | None = None,
        parameters: dict[str, Any] | None = None,
        overwrite: bool = False
    ) -> FeatureMetadata:
        """
        Save features to the store.
        
        Args:
            name: Name of the feature set
            features: Feature DataFrame
            version: Version string (auto-generated if None)
            parameters: Parameters used to generate features
            overwrite: Whether to overwrite existing version
            
        Returns:
            Feature metadata
        """
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        
        # Create directory structure
        feature_dir = self.storage_path / name / version
        
        if feature_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Feature set '{name}' version '{version}' already exists. "
                "Use overwrite=True to replace."
            )
        
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features as parquet
        features_path = feature_dir / "features.parquet"
        features.write_parquet(features_path)
        
        # Create and save metadata
        metadata = FeatureMetadata(
            name=name,
            version=version,
            created_at=datetime.now(),
            features=list(features.columns),
            row_count=len(features),
            source_hash=self._compute_hash(features),
            parameters=parameters or {}
        )
        
        metadata_path = feature_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            f.write(metadata.model_dump_json(indent=2))
        
        # Update cache
        cache_key = f"{name}:{version}"
        if self.cache_enabled:
            self._cache[cache_key] = features
            self._metadata_cache[cache_key] = metadata
        
        logger.info(f"Saved feature set '{name}' version '{version}' ({len(features):,} rows)")
        return metadata
    
    def load(
        self,
        name: str,
        version: str | None = None,
        columns: list[str] | None = None
    ) -> pl.DataFrame:
        """
        Load features from the store.
        
        Args:
            name: Name of the feature set
            version: Version to load (latest if None)
            columns: Specific columns to load (all if None)
            
        Returns:
            Feature DataFrame
        """
        if version is None:
            version = self._get_latest_version(name)
        
        cache_key = f"{name}:{version}"
        
        # Check cache
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Loading '{name}' v{version} from cache")
            features = self._cache[cache_key]
            if columns:
                return features.select(columns)
            return features
        
        # Load from disk
        features_path = self.storage_path / name / version / "features.parquet"
        
        if not features_path.exists():
            raise FileNotFoundError(
                f"Feature set '{name}' version '{version}' not found at {features_path}"
            )
        
        if columns:
            features = pl.read_parquet(features_path, columns=columns)
        else:
            features = pl.read_parquet(features_path)
        
        # Update cache
        if self.cache_enabled:
            self._cache[cache_key] = features
        
        logger.info(f"Loaded feature set '{name}' version '{version}' ({len(features):,} rows)")
        return features
    
    def get_metadata(self, name: str, version: str | None = None) -> FeatureMetadata:
        """Get metadata for a feature set."""
        if version is None:
            version = self._get_latest_version(name)
        
        cache_key = f"{name}:{version}"
        
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        metadata_path = self.storage_path / name / version / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for '{name}' v{version}")
        
        with open(metadata_path) as f:
            metadata = FeatureMetadata.model_validate_json(f.read())
        
        self._metadata_cache[cache_key] = metadata
        return metadata
    
    def list_feature_sets(self) -> list[str]:
        """List all feature set names."""
        return [d.name for d in self.storage_path.iterdir() if d.is_dir()]
    
    def list_versions(self, name: str) -> list[str]:
        """List all versions of a feature set."""
        feature_dir = self.storage_path / name
        if not feature_dir.exists():
            return []
        return sorted([d.name for d in feature_dir.iterdir() if d.is_dir()])
    
    def delete(self, name: str, version: str | None = None) -> None:
        """Delete a feature set or specific version."""
        import shutil
        
        if version:
            path = self.storage_path / name / version
            cache_key = f"{name}:{version}"
        else:
            path = self.storage_path / name
            # Clear all cached versions
            for v in self.list_versions(name):
                cache_key = f"{name}:{v}"
                self._cache.pop(cache_key, None)
                self._metadata_cache.pop(cache_key, None)
        
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted {'version ' + version if version else 'all versions'} of '{name}'")
        
        if version:
            self._cache.pop(f"{name}:{version}", None)
            self._metadata_cache.pop(f"{name}:{version}", None)
    
    def get_point_in_time_features(
        self,
        name: str,
        as_of_date: datetime,
        entity_ids: list[str] | None = None
    ) -> pl.DataFrame:
        """
        Get features as they would have been at a specific point in time.
        
        This is crucial for preventing data leakage during backtesting.
        
        Args:
            name: Feature set name
            as_of_date: Point in time for feature retrieval
            entity_ids: Specific entities to retrieve
            
        Returns:
            Features as of the specified date
        """
        # Find the most recent version before as_of_date
        versions = self.list_versions(name)
        valid_versions = []
        
        for version in versions:
            metadata = self.get_metadata(name, version)
            if metadata.created_at <= as_of_date:
                valid_versions.append((version, metadata.created_at))
        
        if not valid_versions:
            raise ValueError(f"No feature versions found before {as_of_date}")
        
        # Get the latest valid version
        latest_version = max(valid_versions, key=lambda x: x[1])[0]
        
        features = self.load(name, latest_version)
        
        if entity_ids is not None:
            features = features.filter(pl.col("id").is_in(entity_ids))
        
        return features
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._metadata_cache.clear()
        logger.info("Feature store cache cleared")
    
    def _get_latest_version(self, name: str) -> str:
        """Get the latest version of a feature set."""
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(f"No versions found for feature set '{name}'")
        return versions[-1]
    
    def _compute_hash(self, df: pl.DataFrame) -> str:
        """Compute a hash of the DataFrame for lineage tracking."""
        # Use schema and sample of data for hash
        schema_str = str(df.schema)
        sample_str = str(df.head(100).to_pandas().values.tobytes())
        combined = schema_str + sample_str
        return hashlib.md5(combined.encode()).hexdigest()
