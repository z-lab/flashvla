"""Helpers to filter LIBERO-style datasets by task or suite."""

from __future__ import annotations

from typing import Iterable

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


LIBERO_SUITE_TASKS: dict[str, set[str]] = {
    "goal": {
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "put the yellow and white mug in the microwave and close it",
        "turn on the stove and put the moka pot on it",
        "put both the alphabet soup and the cream cheese box in the basket",
        "put both the alphabet soup and the tomato sauce in the basket",
        "put both moka pots on the stove",
        "put both the cream cheese box and the butter in the basket",
        "put the black bowl in the bottom drawer of the cabinet and close it",
        "pick up the book and place it in the back compartment of the caddy",
    },
    "spatial": {
        "put the bowl on the plate",
        "put the wine bottle on the rack",
        "open the top drawer and put the bowl inside",
        "put the cream cheese in the bowl",
        "put the wine bottle on top of the cabinet",
        "push the plate to the front of the stove",
        "turn on the stove",
        "put the bowl on the stove",
        "put the bowl on top of the cabinet",
        "open the middle drawer of the cabinet",
    },
    "object": {
        "pick up the orange juice and place it in the basket",
        "pick up the ketchup and place it in the basket",
        "pick up the cream cheese and place it in the basket",
        "pick up the bbq sauce and place it in the basket",
        "pick up the alphabet soup and place it in the basket",
        "pick up the milk and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        "pick up the butter and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
        "pick up the chocolate pudding and place it in the basket",
    },
    "libero_10": {
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
    },
}


def get_episode_indices_for_tasks(
    ds_meta: LeRobotDatasetMetadata,
    task_descriptions: Iterable[str],
) -> list[int]:
    """Return episode indices whose task description matches any in the set."""
    task_set = set(task_descriptions)
    return [
        ep_idx
        for ep_idx in range(len(ds_meta.episodes))
        if ds_meta.episodes[ep_idx]["tasks"][0] in task_set
    ]


def get_episode_indices_for_suite(
    ds_meta: LeRobotDatasetMetadata,
    suite: str,
) -> list[int]:
    """Return episode indices belonging to a LIBERO suite (goal/spatial/object/libero_10)."""
    if suite not in LIBERO_SUITE_TASKS:
        raise ValueError(f"Unknown suite: {suite!r}. Must be one of {list(LIBERO_SUITE_TASKS)}")
    return get_episode_indices_for_tasks(ds_meta, LIBERO_SUITE_TASKS[suite])


def get_frame_indices_for_episodes(
    ds_meta: LeRobotDatasetMetadata,
    episode_indices: Iterable[int],
) -> list[int]:
    """Return the list of dataset frame indices that belong to the given episodes."""
    frame_indices: list[int] = []
    for ep_idx in episode_indices:
        ep = ds_meta.episodes[ep_idx]
        frame_indices.extend(range(ep["dataset_from_index"], ep["dataset_to_index"]))
    return frame_indices


def get_frame_indices_for_suite(
    ds_meta: LeRobotDatasetMetadata,
    suite: str,
) -> list[int]:
    """Return all dataset frame indices belonging to a LIBERO suite."""
    ep_indices = get_episode_indices_for_suite(ds_meta, suite)
    return get_frame_indices_for_episodes(ds_meta, ep_indices)


def resolve_suite_episodes(
    repo_id: str,
    suite: str | None = None,
    tasks: Iterable[str] | None = None,
) -> list[int] | None:
    """Return the list of episode indices for a LIBERO suite or task set.

    Meant to be passed into ``LeRobotDataset(episodes=...)`` at construction
    time — this is the supported way to filter episodes in LeRobot.

    Returns None when no filter is specified, so callers can do::

        episodes = resolve_suite_episodes(cfg.dataset.repo_id, suite=cfg.train_suite)
        dataset = LeRobotDataset(repo_id, episodes=episodes, ...)
    """
    if suite is None and tasks is None:
        return None

    ds_meta = LeRobotDatasetMetadata(repo_id)
    if tasks is not None:
        return get_episode_indices_for_tasks(ds_meta, tasks)
    return get_episode_indices_for_suite(ds_meta, suite)
