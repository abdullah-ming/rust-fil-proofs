use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use generic_array::typenum::Unsigned;
use log::trace;
use merkletree::merkle;
use merkletree::merkle::{
    get_merkle_tree_leafs, get_merkle_tree_len, is_merkle_tree_size_valid,
    FromIndexedParallelIterator,
};
use merkletree::store::{ExternalReader, Store};
use merkletree::store::{LevelCacheStore, StoreConfig};
use rayon::prelude::*;

use crate::error::*;
use crate::hasher::{Domain, Hasher, PoseidonArity};
use crate::util::{data_at_node, NODE_SIZE};

use super::*;

// Create a DiskTree from the provided config(s), each representing a 'base' layer tree with 'base_tree_len' elements.
pub fn create_disk_tree<Tree: MerkleTreeTrait>(
    base_tree_len: usize,
    configs: &[StoreConfig],
) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>> {
    let base_tree_leafs = get_merkle_tree_leafs(base_tree_len, Tree::Arity::to_usize());

    if Tree::TopTreeArity::to_usize() > 0 {
        ensure!(
            Tree::SubTreeArity::to_usize() > 0,
            "Invalid top arity specified without sub arity"
        );

        DiskTree::from_sub_tree_store_configs(base_tree_leafs, configs)
    } else if Tree::SubTreeArity::to_usize() > 0 {
        ensure!(
            !configs.is_empty(),
            "Cannot create sub-tree with a single tree config"
        );

        DiskTree::from_store_configs(base_tree_leafs, configs)
    } else {
        ensure!(configs.len() == 1, "Invalid tree-shape specified");
        let store = DiskStore::new_from_disk(base_tree_len, Tree::Arity::to_usize(), &configs[0])?;

        DiskTree::from_data_store(store, base_tree_leafs)
    }
}

// Create an LCTree from the provided config(s) and replica(s), each representing a 'base' layer tree with 'base_tree_len' elements.
pub fn create_lc_tree<Tree: MerkleTreeTrait>(
    base_tree_len: usize,
    configs: &[StoreConfig],
    replica_paths: &[PathBuf],
) -> Result<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>> {
    let base_tree_leafs = get_merkle_tree_leafs(base_tree_len, Tree::Arity::to_usize());

    if Tree::TopTreeArity::to_usize() > 0 {
        ensure!(
            Tree::SubTreeArity::to_usize() > 0,
            "Invalid top arity specified without sub arity"
        );

        LCTree::from_sub_tree_store_configs_and_replicas(base_tree_leafs, configs, replica_paths)
    } else if Tree::SubTreeArity::to_usize() > 0 {
        ensure!(
            !configs.is_empty(),
            "Cannot create sub-tree with a single tree config"
        );

        LCTree::from_store_configs_and_replicas(base_tree_leafs, configs, replica_paths)
    } else {
        ensure!(configs.len() == 1, "Invalid tree-shape specified");
        let store = LCStore::new_from_disk_with_reader(
            base_tree_len,
            Tree::Arity::to_usize(),
            &configs[0],
            ExternalReader::new_from_path(&replica_paths[0])?,
        )?;

        LCTree::from_data_store(store, base_tree_leafs)
    }
}

pub fn create_base_merkle_tree<Tree: MerkleTreeTrait>(
    config: Option<StoreConfig>,
    size: usize,
    data: &[u8],
) -> Result<Tree> {
    ensure!(
        data.len() == (NODE_SIZE * size) as usize,
        Error::InvalidMerkleTreeArgs(data.len(), NODE_SIZE, size)
    );

    trace!("create_merkle_tree called with size {}", size);
    trace!(
        "is_merkle_tree_size_valid({}, arity {}) = {}",
        size,
        Tree::Arity::to_usize(),
        is_merkle_tree_size_valid(size, Tree::Arity::to_usize())
    );
    ensure!(
        is_merkle_tree_size_valid(size, Tree::Arity::to_usize()),
        "Invalid merkle tree size given the arity"
    );

    let f = |i| {
        // TODO Replace `expect()` with `context()` (problem is the parallel iterator)
        let d = data_at_node(&data, i).expect("data_at_node math failed");
        // TODO/FIXME: This can panic. FOR NOW, let's leave this since we're experimenting with
        // optimization paths. However, we need to ensure that bad input will not lead to a panic
        // that isn't caught by the FPS API.
        // Unfortunately, it's not clear how to perform this error-handling in the parallel
        // iterator case.
        <Tree::Hasher as Hasher>::Domain::try_from_bytes(d)
            .expect("failed to convert node data to domain element")
    };

    let tree = match config {
        Some(x) => merkle::MerkleTree::<
            <Tree::Hasher as Hasher>::Domain,
            <Tree::Hasher as Hasher>::Function,
            Tree::Store,
            Tree::Arity,
            Tree::SubTreeArity,
            Tree::TopTreeArity,
        >::from_par_iter_with_config((0..size).into_par_iter().map(f), x),
        None => merkle::MerkleTree::<
            <Tree::Hasher as Hasher>::Domain,
            <Tree::Hasher as Hasher>::Function,
            Tree::Store,
            Tree::Arity,
            Tree::SubTreeArity,
            Tree::TopTreeArity,
        >::from_par_iter((0..size).into_par_iter().map(f)),
    }?;

    Ok(Tree::from_merkle(tree))
}

/// Construct a new level cache merkle tree, given the specified
/// config and replica_path.
///
/// Note that while we don't need to pass both the data AND the
/// replica path (since the replica file will contain the same data),
/// we pass both since we have access from all callers and this avoids
/// reading that data from the replica_path here.
pub fn create_lcmerkle_tree<H: Hasher, BaseTreeArity: 'static + PoseidonArity>(
    config: StoreConfig,
    size: usize,
    data: &[u8],
    replica_path: &PathBuf,
) -> Result<LCMerkleTree<H, BaseTreeArity>> {
    trace!("create_lcmerkle_tree called with size {}", size);
    trace!(
        "is_merkle_tree_size_valid({}, arity {}) = {}",
        size,
        BaseTreeArity::to_usize(),
        is_merkle_tree_size_valid(size, BaseTreeArity::to_usize())
    );
    ensure!(
        is_merkle_tree_size_valid(size, BaseTreeArity::to_usize()),
        "Invalid merkle tree size given the arity"
    );
    ensure!(
        data.len() == size * std::mem::size_of::<H::Domain>(),
        "Invalid data length for merkle tree"
    );

    let f = |i| {
        let d = data_at_node(&data, i)?;
        H::Domain::try_from_bytes(d)
    };

    let mut lc_tree: LCMerkleTree<H, BaseTreeArity> =
        LCMerkleTree::<H, BaseTreeArity>::try_from_iter_with_config((0..size).map(f), config)?;

    lc_tree.set_external_reader_path(replica_path)?;

    Ok(lc_tree)
}

/// Open an existing level cache merkle tree, given the specified
/// config and replica_path.
pub fn open_lcmerkle_tree<
    H: Hasher,
    U: 'static + PoseidonArity,
    V: 'static + PoseidonArity,
    W: 'static + PoseidonArity,
>(
    config: StoreConfig,
    size: usize,
    replica_path: &PathBuf,
) -> Result<LCTree<H, U, V, W>> {
    todo!("handle V, W > 0");
    trace!("open_lcmerkle_tree called with size {}", size);
    trace!(
        "is_merkle_tree_size_valid({}, arity {}) = {}",
        size,
        U::to_usize(),
        is_merkle_tree_size_valid(size, U::to_usize())
    );
    ensure!(
        is_merkle_tree_size_valid(size, U::to_usize()),
        "Invalid merkle tree size given the arity"
    );

    let tree_size = get_merkle_tree_len(size, U::to_usize())?;
    let tree_store: LevelCacheStore<H::Domain, _> = LevelCacheStore::new_from_disk_with_reader(
        tree_size,
        U::to_usize(),
        &config,
        ExternalReader::new_from_path(replica_path)?,
    )?;

    ensure!(
        size == get_merkle_tree_leafs(tree_size, U::to_usize()),
        "Inconsistent lcmerkle tree"
    );

    LCTree::from_data_store(tree_store, size)
}

// Given a StoreConfig, generate additional ones with appended numbers
// to uniquely identify them and return the results.  If count is 1,
// the original config is not modified.
pub fn split_config(config: StoreConfig, count: usize) -> Result<Vec<StoreConfig>> {
    if count == 1 {
        return Ok(vec![config]);
    }

    let mut configs = Vec::with_capacity(count);
    for i in 0..count {
        configs.push(StoreConfig::from_config(
            &config,
            format!("{}-{}", config.id, i),
            None,
        ));
        configs[i].levels = config.levels;
    }

    Ok(configs)
}

// Given a StoreConfig, generate additional ones with appended numbers
// to uniquely identify them and return the results.  If count is 1,
// the original config is not modified.
//
// Useful for testing, where there the config may be None.
pub fn split_config_wrapped(
    config: Option<StoreConfig>,
    count: usize,
) -> Result<Vec<Option<StoreConfig>>> {
    if count == 1 {
        return Ok(vec![config]);
    }

    match config {
        Some(c) => {
            let mut configs = Vec::with_capacity(count);
            for i in 0..count {
                configs.push(Some(StoreConfig::from_config(
                    &c,
                    format!("{}-{}", c.id, i),
                    None,
                )));
            }
            Ok(configs)
        }
        None => Ok(vec![None]),
    }
}

// Given a StoreConfig and replica path, append numbers to each to
// uniquely identify them and return the results.  If count is 1, the
// original config and replica path are not modified.
pub fn split_config_and_replica(
    config: StoreConfig,
    replica_path: PathBuf,
    count: usize,
) -> Result<(Vec<StoreConfig>, Vec<PathBuf>)> {
    if count == 1 {
        return Ok((vec![config], vec![replica_path]));
    }

    let mut configs = Vec::with_capacity(count);
    let mut replica_paths = Vec::with_capacity(count);

    for i in 0..count {
        configs.push(StoreConfig::from_config(
            &config,
            format!("{}-{}", config.id, i),
            None,
        ));
        configs[i].levels = config.levels;

        replica_paths.push(
            Path::new(
                format!("{:?}-{}", replica_path, i)
                    .replace("\"", "")
                    .as_str(),
            )
            .to_path_buf(),
        );
    }

    Ok((configs, replica_paths))
}

// FIXME: double check this is correct
pub fn get_base_tree_count<Tree: MerkleTreeTrait>() -> usize {
    if Tree::TopTreeArity::to_usize() == 0 && Tree::SubTreeArity::to_usize() == 0 {
        return 1;
    }

    if Tree::TopTreeArity::to_usize() > 0 {
        Tree::TopTreeArity::to_usize() * Tree::SubTreeArity::to_usize()
    } else {
        Tree::SubTreeArity::to_usize()
    }
}

#[cfg(test)]
pub type ResTree<Tree> = MerkleTreeWrapper<
    <Tree as MerkleTreeTrait>::Hasher,
    <Tree as MerkleTreeTrait>::Store,
    <Tree as MerkleTreeTrait>::Arity,
    <Tree as MerkleTreeTrait>::SubTreeArity,
    <Tree as MerkleTreeTrait>::TopTreeArity,
>;

#[cfg(test)]
use std::io::Write;

#[cfg(test)]
fn generate_base_tree<R: rand::Rng, Tree: MerkleTreeTrait>(
    rng: &mut R,
    nodes: usize,
    temp_path: Option<PathBuf>,
) -> (Vec<u8>, ResTree<Tree>)
where
    Tree::Store: 'static,
{
    let elements = (0..nodes)
        .map(|_| <Tree::Hasher as Hasher>::Domain::random(rng))
        .collect::<Vec<_>>();

    let mut data = Vec::new();
    for el in &elements {
        data.extend_from_slice(AsRef::<[u8]>::as_ref(el));
    }

    if let Some(ref temp_path) = temp_path {
        let id: u64 = rng.gen();
        let replica_path = temp_path.join(format!("replica-path-{}", id));
        let mut f = std::fs::File::create(&replica_path).unwrap();
        f.write_all(&data).unwrap();
        let config = StoreConfig::new(
            &temp_path,
            format!("test-lc-tree-{}", id),
            Tree::Arity::to_usize(),
        );

        let mut tree =
            MerkleTreeWrapper::try_from_iter_with_config(elements.iter().map(|v| (Ok(*v))), config)
                .unwrap();

        {
            // Beware: evil dynamic downcasting RUST MAGIC down below.
            use std::any::Any;

            if let Some(lc_tree) = Any::downcast_mut::<
                merkle::MerkleTree<
                    <Tree::Hasher as Hasher>::Domain,
                    <Tree::Hasher as Hasher>::Function,
                    merkletree::store::LevelCacheStore<
                        <Tree::Hasher as Hasher>::Domain,
                        std::fs::File,
                    >,
                    Tree::Arity,
                    Tree::SubTreeArity,
                    Tree::TopTreeArity,
                >,
            >(&mut tree.inner)
            {
                dbg!("Using LC Tree");
                lc_tree.set_external_reader_path(&replica_path).unwrap();
            }
        }

        (data, tree)
    } else {
        (
            data,
            MerkleTreeWrapper::try_from_iter(elements.iter().map(|v| Ok(*v))).unwrap(),
        )
    }
}

#[cfg(test)]
fn generate_sub_tree<R: rand::Rng, Tree: MerkleTreeTrait>(
    rng: &mut R,
    nodes: usize,
    temp_path: Option<PathBuf>,
) -> (Vec<u8>, ResTree<Tree>)
where
    Tree::Store: 'static,
{
    let base_tree_count = Tree::SubTreeArity::to_usize();
    let base_tree_size = nodes / base_tree_count;
    let mut trees = Vec::with_capacity(base_tree_count);
    let mut data = Vec::new();

    for _ in 0..base_tree_count {
        let (inner_data, tree) = generate_base_tree::<
            R,
            MerkleTreeWrapper<Tree::Hasher, Tree::Store, Tree::Arity>,
        >(rng, base_tree_size, temp_path.clone());
        trees.push(tree);
        data.extend(inner_data);
    }

    (data, MerkleTreeWrapper::from_trees(trees).unwrap())
}

#[cfg(test)]
pub fn generate_tree<Tree: MerkleTreeTrait, R: rand::Rng>(
    rng: &mut R,
    nodes: usize,
    temp_path: Option<PathBuf>,
) -> (Vec<u8>, ResTree<Tree>)
where
    Tree::Store: 'static,
{
    let sub_tree_arity = Tree::SubTreeArity::to_usize();
    let top_tree_arity = Tree::TopTreeArity::to_usize();

    if top_tree_arity > 0 {
        assert!(
            sub_tree_arity != 0,
            "malformed tree with TopTreeArity > 0 and SubTreeARity == 0"
        );

        let mut sub_trees = Vec::with_capacity(top_tree_arity);
        let mut data = Vec::new();
        for _i in 0..top_tree_arity {
            let (inner_data, tree) = generate_sub_tree::<
                R,
                MerkleTreeWrapper<
                    Tree::Hasher,
                    Tree::Store,
                    Tree::Arity,
                    Tree::SubTreeArity,
                    typenum::U0,
                >,
            >(rng, nodes / top_tree_arity, temp_path.clone());

            sub_trees.push(tree);
            data.extend(inner_data);
        }
        (data, MerkleTreeWrapper::from_sub_trees(sub_trees).unwrap())
    } else {
        if sub_tree_arity > 0 {
            generate_sub_tree::<R, Tree>(rng, nodes, temp_path)
        } else {
            generate_base_tree::<R, Tree>(rng, nodes, temp_path)
        }
    }
}
