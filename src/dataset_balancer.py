"""
Dataset Balancing Strategies for Multi-Dataset Training
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config


class DatasetBalancer:
    """
    Balance datasets across multiple emotion recognition datasets.
    
    Supports multiple balancing strategies to ensure fair representation
    of datasets and emotions in training data.
    """
    
    def __init__(
        self,
        strategy: str = "stratified_per_dataset",
        max_samples_per_dataset: Optional[int] = None,
        min_samples_per_emotion: int = 100,
        random_state: int = None,
    ):
        """
        Initialize dataset balancer.
        
        Args:
            strategy: Balancing strategy to use
            max_samples_per_dataset: Maximum samples per dataset (for proportional strategy)
            min_samples_per_emotion: Minimum samples per emotion class
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.max_samples_per_dataset = max_samples_per_dataset
        self.min_samples_per_emotion = min_samples_per_emotion
        self.random_state = random_state or config.RANDOM_SEED
        
        if strategy not in ["equal_per_dataset", "equal_per_emotion", "proportional", "stratified_per_dataset"]:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                "Must be one of: equal_per_dataset, equal_per_emotion, proportional, stratified_per_dataset"
            )
    
    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply balancing strategy to dataset.
        
        Args:
            df: Combined DataFrame from all datasets with 'dataset' and 'emotion' columns
            
        Returns:
            Balanced DataFrame
        """
        if self.strategy == "equal_per_dataset":
            return self._equal_per_dataset(df)
        elif self.strategy == "equal_per_emotion":
            return self._equal_per_emotion(df)
        elif self.strategy == "proportional":
            return self._proportional(df)
        elif self.strategy == "stratified_per_dataset":
            return self._stratified_per_dataset(df)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _equal_per_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy 1: Equal samples per dataset.
        
        Sample equal number from each dataset, preserving emotion distribution within each.
        """
        datasets = df["dataset"].unique()
        min_size = df.groupby("dataset").size().min()
        
        balanced_dfs = []
        for dataset in datasets:
            dataset_df = df[df["dataset"] == dataset]
            # Sample equal number, stratified by emotion
            if len(dataset_df) > min_size:
                sampled = dataset_df.groupby("emotion", group_keys=False).apply(
                    lambda x: x.sample(
                        min(len(x), int(min_size / len(df["emotion"].unique()))),
                        random_state=self.random_state
                    )
                )
                # If we need more samples, sample randomly
                if len(sampled) < min_size:
                    remaining = dataset_df[~dataset_df.index.isin(sampled.index)]
                    needed = min_size - len(sampled)
                    if len(remaining) > 0:
                        additional = remaining.sample(
                            min(needed, len(remaining)),
                            random_state=self.random_state
                        )
                        sampled = pd.concat([sampled, additional])
                balanced_dfs.append(sampled)
            else:
                balanced_dfs.append(dataset_df)
        
        result = pd.concat(balanced_dfs, ignore_index=True)
        return result
    
    def _equal_per_emotion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy 2: Equal samples per emotion.
        
        Sample equal number per emotion across all datasets.
        If min_samples_per_emotion is set, uses that as the target size.
        """
        emotions = df["emotion"].unique()
        min_size = df.groupby("emotion").size().min()
        
        # If min_samples_per_emotion is set and is less than min_size, use it
        # This allows capping all emotions to a specific size
        if self.min_samples_per_emotion and self.min_samples_per_emotion < min_size:
            target_size = self.min_samples_per_emotion
        else:
            target_size = min_size
        
        balanced_dfs = []
        for emotion in emotions:
            emotion_df = df[df["emotion"] == emotion]
            if len(emotion_df) > target_size:
                sampled = emotion_df.sample(
                    target_size,
                    random_state=self.random_state
                )
            else:
                sampled = emotion_df
            balanced_dfs.append(sampled)
        
        result = pd.concat(balanced_dfs, ignore_index=True)
        return result
    
    def _proportional(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy 3: Proportional with limits.
        
        Maintain natural dataset ratios but cap maximum per dataset.
        """
        if self.max_samples_per_dataset is None:
            # No limit, return as-is
            return df
        
        balanced_dfs = []
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            if len(dataset_df) > self.max_samples_per_dataset:
                # Sample proportionally by emotion
                emotion_counts = dataset_df["emotion"].value_counts()
                total = len(dataset_df)
                
                sampled_parts = []
                for emotion in dataset_df["emotion"].unique():
                    emotion_df = dataset_df[dataset_df["emotion"] == emotion]
                    proportion = len(emotion_df) / total
                    target_size = int(self.max_samples_per_dataset * proportion)
                    target_size = max(1, target_size)  # At least 1 sample
                    
                    if len(emotion_df) > target_size:
                        sampled = emotion_df.sample(
                            target_size,
                            random_state=self.random_state
                        )
                    else:
                        sampled = emotion_df
                    sampled_parts.append(sampled)
                
                sampled = pd.concat(sampled_parts, ignore_index=True)
                # If still over limit, sample randomly
                if len(sampled) > self.max_samples_per_dataset:
                    sampled = sampled.sample(
                        self.max_samples_per_dataset,
                        random_state=self.random_state
                    )
                balanced_dfs.append(sampled)
            else:
                balanced_dfs.append(dataset_df)
        
        result = pd.concat(balanced_dfs, ignore_index=True)
        return result
    
    def _stratified_per_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy 4: Stratified per dataset (RECOMMENDED).
        
        Create stratified splits within each dataset, preserving emotion distribution.
        This is the most robust strategy for generalization.
        """
        # For this strategy, we don't balance here - we just ensure
        # each dataset has sufficient samples per emotion
        # Actual balancing happens during split creation
        
        # Filter out datasets/emotions with too few samples
        emotion_counts = df.groupby(["dataset", "emotion"]).size()
        valid_indices = []
        
        for (dataset, emotion), count in emotion_counts.items():
            if count >= self.min_samples_per_emotion:
                mask = (df["dataset"] == dataset) & (df["emotion"] == emotion)
                valid_indices.extend(df[mask].index.tolist())
        
        result = df.loc[valid_indices].copy()
        
        if len(result) < len(df):
            print(
                f"⚠ Filtered {len(df) - len(result)} samples with insufficient "
                f"representation (<{self.min_samples_per_emotion} per emotion per dataset)"
            )
        
        return result
    
    def create_stratified_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/val/test splits.
        
        For "stratified_per_dataset" strategy, creates splits within each dataset
        and combines them. For other strategies, creates global stratified splits.
        
        Args:
            df: DataFrame to split
            train_ratio: Training set ratio (defaults to config.TRAIN_RATIO)
            val_ratio: Validation set ratio (defaults to config.VAL_RATIO)
            test_ratio: Test set ratio (defaults to config.TEST_RATIO)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_ratio = train_ratio or config.TRAIN_RATIO
        val_ratio = val_ratio or config.VAL_RATIO
        test_ratio = test_ratio or config.TEST_RATIO
        
        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        if self.strategy == "stratified_per_dataset":
            return self._create_stratified_splits_per_dataset(df, train_ratio, val_ratio, test_ratio)
        else:
            return self._create_global_stratified_splits(df, train_ratio, val_ratio, test_ratio)
    
    def _create_stratified_splits_per_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified splits within each dataset, then combine.
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            
            # First split: train vs (val+test)
            train_size = train_ratio
            temp_size = val_ratio + test_ratio
            
            train_temp, val_test = train_test_split(
                dataset_df,
                test_size=temp_size,
                stratify=dataset_df["emotion"],
                random_state=self.random_state,
            )
            
            # Second split: val vs test
            val_size = val_ratio / temp_size
            val_temp, test_temp = train_test_split(
                val_test,
                test_size=(1 - val_size),
                stratify=val_test["emotion"],
                random_state=self.random_state,
            )
            
            train_dfs.append(train_temp)
            val_dfs.append(val_temp)
            test_dfs.append(test_temp)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        return train_df, val_df, test_df
    
    def _create_global_stratified_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create global stratified splits across all datasets.
        """
        # First split: train vs (val+test)
        train_size = train_ratio
        temp_size = val_ratio + test_ratio
        
        train_temp, val_test = train_test_split(
            df,
            test_size=temp_size,
            stratify=df["emotion"],
            random_state=self.random_state,
        )
        
        # Second split: val vs test
        val_size = val_ratio / temp_size
        val_temp, test_temp = train_test_split(
            val_test,
            test_size=(1 - val_size),
            stratify=val_test["emotion"],
            random_state=self.random_state,
        )
        
        return train_temp, val_temp, test_temp
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get balancing statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": len(df),
            "datasets": df["dataset"].value_counts().to_dict(),
            "emotions": df["emotion"].value_counts().to_dict(),
            "dataset_emotion_matrix": df.groupby(["dataset", "emotion"]).size().to_dict(),
        }
        
        return stats


__all__ = ["DatasetBalancer"]

