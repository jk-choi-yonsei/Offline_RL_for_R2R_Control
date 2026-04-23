"""
MDP Dataset construction for d3rlpy.
Converts preprocessed data into d3rlpy-compatible MDPDataset format.
"""

import os
import logging
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

from src.data.config import DataConfig, RESULTS_DIR

logger = logging.getLogger(__name__)


def split_data(
    mdp_data: Dict[str, np.ndarray],
    config: Optional[DataConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split MDP data into train/val/test sets.
    Splits by episode boundaries (terminal flags) to avoid data leakage.

    Args:
        mdp_data: Dictionary with observations, actions, rewards, etc.
        config: DataConfig instance.

    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    if config is None:
        config = DataConfig()

    np.random.seed(config.random_seed)

    # Find episode boundaries
    terminals = mdp_data["terminals"]
    episode_ends = np.where(terminals == 1.0)[0]

    if len(episode_ends) == 0:
        # No terminal flags - split randomly
        n = len(mdp_data["observations"])
        indices = np.random.permutation(n)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
    else:
        # Split by episodes
        episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
        n_episodes = len(episode_starts)

        ep_indices = np.random.permutation(n_episodes)
        train_end = int(n_episodes * config.train_ratio)
        val_end = int(n_episodes * (config.train_ratio + config.val_ratio))

        train_episodes = ep_indices[:train_end]
        val_episodes = ep_indices[train_end:val_end]
        test_episodes = ep_indices[val_end:]

        def get_indices(episodes):
            idx = []
            for ep in episodes:
                start = episode_starts[ep]
                end = episode_ends[ep] + 1
                idx.extend(range(start, end))
            return np.array(idx, dtype=int)

        train_idx = get_indices(train_episodes)
        val_idx = get_indices(val_episodes)
        test_idx = get_indices(test_episodes)

    def subset(data, idx):
        return {k: v[idx] for k, v in data.items()}

    train_data = subset(mdp_data, train_idx)
    val_data = subset(mdp_data, val_idx)
    test_data = subset(mdp_data, test_idx)

    logger.info(
        f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_data, val_data, test_data


def chrono_split_data(
    mdp_data: Dict[str, np.ndarray],
    config: Optional[DataConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Chronological split by lot file_id (temporal order).
    Early lots -> train, middle -> val, late lots -> test.

    Requires mdp_data to contain 'file_ids' (added by preprocess_cmp1_r2r).
    This simulates distribution shift: model trained on low-wear lots,
    tested on high-wear lots where DriftEncoder context is crucial.

    Args:
        mdp_data: Dictionary including 'file_ids' key.
        config: DataConfig instance.

    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    if config is None:
        config = DataConfig()

    if "file_ids" not in mdp_data:
        raise ValueError("mdp_data must contain 'file_ids' for chronological split. "
                         "Ensure preprocess_cmp1_r2r was run with file_id tracking.")

    file_ids = mdp_data["file_ids"]
    unique_fids = np.sort(np.unique(file_ids))
    n_fids = len(unique_fids)

    train_end = int(n_fids * config.train_ratio)
    val_end = int(n_fids * (config.train_ratio + config.val_ratio))

    train_fids = set(unique_fids[:train_end].tolist())
    val_fids = set(unique_fids[train_end:val_end].tolist())
    test_fids = set(unique_fids[val_end:].tolist())

    train_idx = np.where(np.isin(file_ids, list(train_fids)))[0]
    val_idx = np.where(np.isin(file_ids, list(val_fids)))[0]
    test_idx = np.where(np.isin(file_ids, list(test_fids)))[0]

    def subset(data, idx):
        return {k: v[idx] for k, v in data.items()}

    train_data = subset(mdp_data, train_idx)
    val_data = subset(mdp_data, val_idx)
    test_data = subset(mdp_data, test_idx)

    logger.info(
        f"Chrono split: {len(train_fids)} train lots / {len(val_fids)} val lots / "
        f"{len(test_fids)} test lots (file_id {unique_fids[val_end]} - {unique_fids[-1]} in test)"
    )
    logger.info(
        f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_data, val_data, test_data


def create_d3rlpy_dataset(mdp_data: Dict[str, np.ndarray]):
    """
    Create d3rlpy MDPDataset from preprocessed data.

    Args:
        mdp_data: Dictionary with observations, actions, rewards,
                  next_observations, terminals.

    Returns:
        d3rlpy MDPDataset instance.
    """
    try:
        import d3rlpy

        dataset = d3rlpy.dataset.MDPDataset(
            observations=mdp_data["observations"],
            actions=mdp_data["actions"],
            rewards=mdp_data["rewards"],
            terminals=mdp_data["terminals"],
        )
        logger.info(f"Created d3rlpy MDPDataset: {dataset.size()} transitions")
        return dataset
    except ImportError:
        logger.warning(
            "d3rlpy not installed. Returning raw dict. "
            "Install with: pip install d3rlpy"
        )
        return mdp_data


def save_dataset(
    mdp_data: Dict[str, np.ndarray],
    scalers: Dict,
    target_rr: Dict,
    name: str = "cmp2",
):
    """
    Save preprocessed dataset and metadata.

    Args:
        mdp_data: MDP data dictionary.
        scalers: Scaler objects for state/action normalization.
        target_rr: Target removal rate information.
        name: Dataset name for file naming.
    """
    save_dir = os.path.join(RESULTS_DIR, "data")
    os.makedirs(save_dir, exist_ok=True)

    # Save numpy arrays
    for key, arr in mdp_data.items():
        np.save(os.path.join(save_dir, f"{name}_{key}.npy"), arr)

    # Save metadata
    metadata = {
        "scalers": scalers,
        "target_rr": target_rr,
        "state_dim": mdp_data["observations"].shape[1],
        "action_dim": mdp_data["actions"].shape[1],
        "n_transitions": len(mdp_data["observations"]),
    }
    with open(os.path.join(save_dir, f"{name}_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Dataset saved to {save_dir}/{name}_*.npy")


def load_dataset(name: str = "cmp2") -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load saved preprocessed dataset.

    Args:
        name: Dataset name.

    Returns:
        Tuple of (mdp_data, metadata).
    """
    save_dir = os.path.join(RESULTS_DIR, "data")

    mdp_data = {}
    for key in ["observations", "actions", "rewards", "next_observations", "terminals"]:
        fpath = os.path.join(save_dir, f"{name}_{key}.npy")
        if os.path.exists(fpath):
            mdp_data[key] = np.load(fpath)

    with open(os.path.join(save_dir, f"{name}_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    logger.info(
        f"Loaded dataset '{name}': {metadata['n_transitions']} transitions, "
        f"state_dim={metadata['state_dim']}, action_dim={metadata['action_dim']}"
    )
    return mdp_data, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.data.preprocess_cmp2 import preprocess_cmp2

    # Preprocess CMP2
    mdp_data, targets, s_scaler, a_scaler = preprocess_cmp2()

    # Split
    train, val, test = split_data(mdp_data)

    # Save
    save_dataset(
        train,
        scalers={"state": s_scaler, "action": a_scaler},
        target_rr=targets,
        name="cmp2_train",
    )
    save_dataset(val, scalers={}, target_rr=targets, name="cmp2_val")
    save_dataset(test, scalers={}, target_rr=targets, name="cmp2_test")

    # Create d3rlpy dataset
    d3rlpy_dataset = create_d3rlpy_dataset(train)

    print("\n" + "=" * 60)
    print("MDP Dataset Construction Complete")
    print("=" * 60)
    print(f"Train: {len(train['observations'])} transitions")
    print(f"Val:   {len(val['observations'])} transitions")
    print(f"Test:  {len(test['observations'])} transitions")
