#![allow(clippy::len_without_is_empty)]

use std::marker::PhantomData;
use std::path::PathBuf;

use anyhow::{ensure, Result};
use generic_array::typenum::{self, Unsigned, U0, U2, U4, U8};
use log::trace;
use merkletree::hash::Algorithm;
use merkletree::hash::Hashable;
use merkletree::merkle;
use merkletree::merkle::{
    get_merkle_tree_leafs, get_merkle_tree_len, is_merkle_tree_size_valid,
    FromIndexedParallelIterator,
};
use merkletree::proof;
use merkletree::store::{LevelCacheStore, StoreConfig};
use paired::bls12_381::Fr;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::drgraph::graph_height;
use crate::error::*;
use crate::hasher::{Domain, Hasher, PoseidonArity};
use crate::util::{data_at_node, NODE_SIZE};

// FIXME: Move from filecoin-proofs/src/constants to here?
pub const SECTOR_SIZE_2_KIB: u64 = 2_048;
pub const SECTOR_SIZE_8_MIB: u64 = 1 << 23;
pub const SECTOR_SIZE_512_MIB: u64 = 1 << 29;
pub const SECTOR_SIZE_32_GIB: u64 = 1 << 35;

pub const SECTOR_SIZE_4_KIB: u64 = 2 * SECTOR_SIZE_2_KIB;
pub const SECTOR_SIZE_16_MIB: u64 = 2 * SECTOR_SIZE_8_MIB;
pub const SECTOR_SIZE_1_GIB: u64 = 2 * SECTOR_SIZE_512_MIB;
pub const SECTOR_SIZE_64_GIB: u64 = 2 * SECTOR_SIZE_32_GIB;

// FIXME: Unsupported size, but used for quickly testing a top level
// tree consisting of 8x 4 KIB trees (each consisting of 2x 2 KIB trees)
pub const SECTOR_SIZE_32_KIB: u64 = 8 * SECTOR_SIZE_4_KIB;

// Reexport here, so we don't depend on merkletree directly in other places.
pub use merkletree::store::{ExternalReader, Store};

pub type DiskStore<E> = merkletree::store::DiskStore<E>;

pub type MerkleStore<T> = DiskStore<T>;

pub type DiskTree<H, U, V, W> = MerkleTreeWrapper<H, DiskStore<<H as Hasher>::Domain>, U, V, W>;
pub type LCTree<H, U, V, W> =
    MerkleTreeWrapper<H, LevelCacheStore<<H as Hasher>::Domain, std::fs::File>, U, V, W>;

pub type MerkleTree<H, U> = DiskTree<H, U, U0, U0>;
pub type LCMerkleTree<H, U> = LCTree<H, U, U0, U0>;

pub type BinaryMerkleTree<H> = MerkleTree<H, U2>;
pub type BinaryLCMerkleTree<H> = LCMerkleTree<H, U2>;

pub type BinarySubMerkleTree<H> = DiskTree<H, U2, U2, U0>;

pub type QuadMerkleTree<H> = MerkleTree<H, U4>;
pub type QuadLCMerkleTree<H> = LCMerkleTree<H, U4>;

pub type OctMerkleTree<H> = DiskTree<H, U8, U0, U0>;
pub type OctSubMerkleTree<H> = DiskTree<H, U8, U2, U0>;
pub type OctTopMerkleTree<H> = DiskTree<H, U8, U8, U2>;

pub type OctLCMerkleTree<H> = LCTree<H, U8, U0, U0>;
pub type OctLCSubMerkleTree<H> = LCTree<H, U8, U2, U0>;
pub type OctLCTopMerkleTree<H> = LCTree<H, U8, U8, U2>;

pub trait MerkleTreeTrait: Send + Sync + std::fmt::Debug {
    type Arity: 'static + PoseidonArity;
    type SubTreeArity: 'static + PoseidonArity;
    type TopTreeArity: 'static + PoseidonArity;
    type Hasher: Hasher;
    type Store: Store<<Self::Hasher as Hasher>::Domain>;
    type Proof: MerkleProofTrait<
        Hasher = Self::Hasher,
        Arity = Self::Arity,
        SubTreeArity = Self::SubTreeArity,
        TopTreeArity = Self::TopTreeArity,
    >;

    fn display() -> String;
    fn root(&self) -> <Self::Hasher as Hasher>::Domain;
    fn gen_proof(&self, i: usize) -> Result<Self::Proof>;
    fn gen_cached_proof(&self, i: usize, levels: usize) -> Result<Self::Proof>;
    fn from_merkle(
        tree: merkle::MerkleTree<
            <Self::Hasher as Hasher>::Domain,
            <Self::Hasher as Hasher>::Function,
            Self::Store,
            Self::Arity,
            Self::SubTreeArity,
            Self::TopTreeArity,
        >,
    ) -> Self;
    fn height(&self) -> usize;
    fn leaves(&self) -> usize;
}

pub trait MerkleProofTrait:
    Clone + Serialize + serde::de::DeserializeOwned + std::fmt::Debug + Sync + Send
{
    type Hasher: Hasher;
    type Arity: 'static + PoseidonArity;
    type SubTreeArity: 'static + PoseidonArity;
    type TopTreeArity: 'static + PoseidonArity;

    fn new_from_proof(
        p: &proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
    ) -> Result<Self>;

    fn as_options(&self) -> Vec<(Vec<Option<Fr>>, Option<usize>)> {
        self.path()
            .iter()
            .map(|v| {
                (
                    v.0.iter().copied().map(Into::into).map(Some).collect(),
                    Some(v.1),
                )
            })
            .collect::<Vec<_>>()
    }

    fn into_options_with_leaf(self) -> (Option<Fr>, Vec<(Vec<Option<Fr>>, Option<usize>)>) {
        let leaf = self.leaf();
        let path = self.path();
        (
            Some(leaf.into()),
            path.into_iter()
                .map(|(a, b)| {
                    (
                        a.iter().copied().map(Into::into).map(Some).collect(),
                        Some(b),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
    fn as_pairs(&self) -> Vec<(Vec<Fr>, usize)> {
        self.path()
            .iter()
            .map(|v| (v.0.iter().copied().map(Into::into).collect(), v.1))
            .collect::<Vec<_>>()
    }
    fn verify(&self) -> bool;
    fn validate(&self, node: usize) -> bool {
        node == self.path_index()
    }
    fn validate_data(&self, data: <Self::Hasher as Hasher>::Domain) -> bool {
        self.leaf() == data
    }
    fn leaf(&self) -> <Self::Hasher as Hasher>::Domain;
    fn root(&self) -> &<Self::Hasher as Hasher>::Domain;
    fn len(&self) -> usize;
    fn path(&self) -> Vec<(Vec<<Self::Hasher as Hasher>::Domain>, usize)>;
    fn path_index(&self) -> usize {
        self.path()
            .iter()
            .rev()
            .fold(0, |acc, (_, index)| (acc * Self::Arity::to_usize()) + index)
    }
    fn proves_challenge(&self, challenge: usize) -> bool;

    /// Calcluates the exected length of the full path, given the number of leaves in the base layer.
    fn expected_len(&self, leaves: usize) -> usize {
        compound_path_length::<Self::Arity, Self::SubTreeArity, Self::TopTreeArity>(leaves)
    }
}

pub fn compound_path_length<A: Unsigned, B: Unsigned, C: Unsigned>(leaves: usize) -> usize {
    let leaves = if C::to_usize() > 0 {
        leaves / C::to_usize() / B::to_usize()
    } else if B::to_usize() > 0 {
        leaves / B::to_usize()
    } else {
        leaves
    };

    let mut len = graph_height::<A>(leaves) - 1;

    if B::to_usize() > 0 {
        len += 1;
    }

    if C::to_usize() > 0 {
        len += 1;
    }

    len
}
pub fn compound_tree_height<A: Unsigned, B: Unsigned, C: Unsigned>(leaves: usize) -> usize {
    // base layer
    let a = graph_height::<A>(leaves) - 1;

    // sub tree layer
    let b = if B::to_usize() > 0 {
        B::to_usize() - 1
    } else {
        0
    };

    // top tree layer
    let c = if C::to_usize() > 0 {
        C::to_usize() - 1
    } else {
        0
    };

    a + b + c
}

impl<
        H: Hasher,
        S: Store<<H as Hasher>::Domain>,
        U: 'static + PoseidonArity,
        V: 'static + PoseidonArity,
        W: 'static + PoseidonArity,
    > MerkleTreeTrait for MerkleTreeWrapper<H, S, U, V, W>
{
    type Arity = U;
    type SubTreeArity = V;
    type TopTreeArity = W;
    type Hasher = H;
    type Store = S;
    type Proof = MerkleProof<Self::Hasher, Self::Arity, Self::SubTreeArity, Self::TopTreeArity>;

    fn display() -> String {
        format!("MerkleTree<{}>", U::to_usize())
    }

    fn root(&self) -> <Self::Hasher as Hasher>::Domain {
        self.inner.root()
    }

    fn gen_proof(&self, i: usize) -> Result<Self::Proof> {
        let proof = self.inner.gen_proof(i)?;

        // For development and debugging.
        assert!(proof.validate::<H::Function>().unwrap());

        MerkleProof::new_from_proof(&proof)
    }

    fn gen_cached_proof(&self, i: usize, levels: usize) -> Result<Self::Proof> {
        let proof = self.inner.gen_cached_proof(i, levels)?;
        MerkleProof::new_from_proof(&proof)
    }

    fn from_merkle(
        tree: merkle::MerkleTree<
            <Self::Hasher as Hasher>::Domain,
            <Self::Hasher as Hasher>::Function,
            Self::Store,
            Self::Arity,
            Self::SubTreeArity,
            Self::TopTreeArity,
        >,
    ) -> Self {
        MerkleTreeWrapper {
            inner: tree,
            h: Default::default(),
        }
    }

    fn height(&self) -> usize {
        self.inner.height()
    }

    fn leaves(&self) -> usize {
        self.inner.leafs()
    }
}

macro_rules! forward_method {
    ($caller:expr, $name:ident) => {
        match $caller {
            ProofData::Single(ref proof) => proof.$name(),
            ProofData::Sub(ref proof) => proof.$name(),
            ProofData::Top(ref proof) => proof.$name(),
        }
    };
    ($caller:expr, $name:ident, $( $args:expr ),+) => {
        match $caller {
            ProofData::Single(ref proof) => proof.$name($($args),+),
            ProofData::Sub(ref proof) => proof.$name($($args),+),
            ProofData::Top(ref proof) => proof.$name($($args),+),
        }
    };
}

impl<
        H: Hasher,
        Arity: 'static + PoseidonArity,
        SubTreeArity: 'static + PoseidonArity,
        TopTreeArity: 'static + PoseidonArity,
    > MerkleProofTrait for MerkleProof<H, Arity, SubTreeArity, TopTreeArity>
{
    type Hasher = H;
    type Arity = Arity;
    type SubTreeArity = SubTreeArity;
    type TopTreeArity = TopTreeArity;

    fn new_from_proof(
        p: &proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
    ) -> Result<Self> {
        if p.top_layer_nodes() > 0 {
            Ok(MerkleProof {
                data: ProofData::Top(TopProof::new_from_proof(p)?),
            })
        } else if p.sub_layer_nodes() > 0 {
            Ok(MerkleProof {
                data: ProofData::Sub(SubProof::new_from_proof(p)?),
            })
        } else {
            Ok(MerkleProof {
                data: ProofData::Single(SingleProof::new_from_proof(p)?),
            })
        }
    }

    fn verify(&self) -> bool {
        forward_method!(self.data, verify)
    }

    fn leaf(&self) -> H::Domain {
        forward_method!(self.data, leaf)
    }

    fn root(&self) -> &H::Domain {
        forward_method!(self.data, root)
    }

    fn len(&self) -> usize {
        forward_method!(self.data, len)
    }

    fn path(&self) -> Vec<(Vec<H::Domain>, usize)> {
        forward_method!(self.data, path)
    }
    fn path_index(&self) -> usize {
        forward_method!(self.data, path_index)
    }

    fn proves_challenge(&self, challenge: usize) -> bool {
        forward_method!(self.data, proves_challenge, challenge)
    }
}

pub struct MerkleTreeWrapper<
    H: Hasher,
    S: Store<<H as Hasher>::Domain>,
    U: PoseidonArity,
    V: PoseidonArity = typenum::U0,
    W: PoseidonArity = typenum::U0,
> {
    pub inner: merkle::MerkleTree<<H as Hasher>::Domain, <H as Hasher>::Function, S, U, V, W>,
    pub h: PhantomData<H>,
}

impl<
        H: Hasher,
        S: Store<<H as Hasher>::Domain>,
        U: PoseidonArity,
        V: PoseidonArity,
        W: PoseidonArity,
    > MerkleTreeWrapper<H, S, U, V, W>
{
    pub fn from_merkle(
        tree: merkle::MerkleTree<<H as Hasher>::Domain, <H as Hasher>::Function, S, U, V, W>,
    ) -> Self {
        Self {
            inner: tree,
            h: Default::default(),
        }
    }

    pub fn new<I: IntoIterator<Item = H::Domain>>(data: I) -> Result<Self> {
        let tree = merkle::MerkleTree::new(data)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn new_with_config<I: IntoIterator<Item = H::Domain>>(
        data: I,
        config: StoreConfig,
    ) -> Result<Self> {
        let tree = merkle::MerkleTree::new_with_config(data, config)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_data_with_config<O: Hashable<H::Function>, I: IntoIterator<Item = O>>(
        data: I,
        config: StoreConfig,
    ) -> Result<Self> {
        let tree = merkle::MerkleTree::from_data_with_config(data, config)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_data_store(data: S, leafs: usize) -> Result<Self> {
        let tree = merkle::MerkleTree::from_data_store(data, leafs)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_tree_slice(data: &[u8], leafs: usize) -> Result<Self> {
        let tree = merkle::MerkleTree::from_tree_slice(data, leafs)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_tree_slice_with_config(
        data: &[u8],
        leafs: usize,
        config: StoreConfig,
    ) -> Result<Self> {
        let tree = merkle::MerkleTree::from_tree_slice_with_config(data, leafs, config)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_trees(trees: Vec<MerkleTreeWrapper<H, S, U, U0, U0>>) -> Result<Self> {
        let trees = trees.into_iter().map(|t| t.inner).collect();
        let tree = merkle::MerkleTree::from_trees(trees)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_sub_trees(trees: Vec<MerkleTreeWrapper<H, S, U, V, U0>>) -> Result<Self> {
        let trees = trees.into_iter().map(|t| t.inner).collect();
        let tree = merkle::MerkleTree::from_sub_trees(trees)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_sub_trees_as_trees(trees: Vec<MerkleTreeWrapper<H, S, U, U0, U0>>) -> Result<Self> {
        let trees = trees.into_iter().map(|t| t.inner).collect();
        let tree = merkle::MerkleTree::from_sub_trees_as_trees(trees)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_slices(
        tree_data: &[&[u8]],
        leafs: usize,
    ) -> Result<MerkleTreeWrapper<H, S, U, V, U0>> {
        let tree = merkle::MerkleTree::<
                <H as Hasher>::Domain, <H as Hasher>::Function, S, U, V, U0
        >::from_slices(tree_data, leafs)?;
        Ok(MerkleTreeWrapper::from_merkle(tree))
    }

    pub fn from_slices_with_configs(
        tree_data: &[&[u8]],
        leafs: usize,
        configs: &[StoreConfig],
    ) -> Result<Self> {
        let tree = merkle::MerkleTree::from_slices_with_configs(tree_data, leafs, configs)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_stores(leafs: usize, stores: Vec<S>) -> Result<Self> {
        let tree = merkle::MerkleTree::from_stores(leafs, stores)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_store_configs(leafs: usize, configs: &[StoreConfig]) -> Result<Self> {
        let tree = merkle::MerkleTree::from_store_configs(leafs, configs)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_store_configs_and_replicas(
        leafs: usize,
        configs: &[StoreConfig],
        replica_paths: &[PathBuf],
    ) -> Result<LCTree<H, U, V, W>> {
        let tree =
            merkle::MerkleTree::from_store_configs_and_replicas(leafs, configs, replica_paths)?;
        Ok(LCTree::from_merkle(tree))
    }

    pub fn from_sub_tree_store_configs(leafs: usize, configs: &[StoreConfig]) -> Result<Self> {
        let tree = merkle::MerkleTree::from_sub_tree_store_configs(leafs, configs)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn try_from_iter<I: IntoIterator<Item = Result<H::Domain>>>(into: I) -> Result<Self> {
        let tree = merkle::MerkleTree::try_from_iter(into)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_sub_tree_store_configs_and_replicas(
        leafs: usize,
        configs: &[StoreConfig],
        replica_paths: &[PathBuf],
    ) -> Result<LCTree<H, U, V, W>> {
        let tree = merkle::MerkleTree::from_sub_tree_store_configs_and_replicas(
            leafs,
            configs,
            replica_paths,
        )?;
        Ok(LCTree::from_merkle(tree))
    }

    pub fn try_from_iter_with_config<I: IntoIterator<Item = Result<H::Domain>>>(
        into: I,
        config: StoreConfig,
    ) -> Result<Self> {
        let tree = merkle::MerkleTree::try_from_iter_with_config(into, config)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_par_iter<I>(par_iter: I) -> Result<Self>
    where
        I: IntoParallelIterator<Item = H::Domain>,
        I::Iter: IndexedParallelIterator,
    {
        let tree = merkle::MerkleTree::from_par_iter(par_iter)?;
        Ok(Self::from_merkle(tree))
    }

    pub fn from_par_iter_with_config<I>(par_iter: I, config: StoreConfig) -> Result<Self>
    where
        I: IntoParallelIterator<Item = H::Domain>,
        I::Iter: IndexedParallelIterator,
    {
        let tree = merkle::MerkleTree::from_par_iter_with_config(par_iter, config)?;
        Ok(Self::from_merkle(tree))
    }
}

impl<
        H: Hasher,
        S: Store<<H as Hasher>::Domain>,
        BaseArity: PoseidonArity,
        SubTreeArity: PoseidonArity,
        TopTreeArity: PoseidonArity,
    > std::fmt::Debug for MerkleTreeWrapper<H, S, BaseArity, SubTreeArity, TopTreeArity>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MerkleTreeWrapper")
            .field("inner", &self.inner)
            .field("Hasher", &H::name())
            .finish()
    }
}

impl<
        H: Hasher,
        S: Store<<H as Hasher>::Domain>,
        BaseArity: PoseidonArity,
        SubTreeArity: PoseidonArity,
        TopTreeArity: PoseidonArity,
    > std::ops::Deref for MerkleTreeWrapper<H, S, BaseArity, SubTreeArity, TopTreeArity>
{
    type Target =
        merkle::MerkleTree<H::Domain, H::Function, S, BaseArity, SubTreeArity, TopTreeArity>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<
        H: Hasher,
        S: Store<<H as Hasher>::Domain>,
        BaseArity: PoseidonArity,
        SubTreeArity: PoseidonArity,
        TopTreeArity: PoseidonArity,
    > std::ops::DerefMut for MerkleTreeWrapper<H, S, BaseArity, SubTreeArity, TopTreeArity>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug)]
pub enum OctTreeData<H: Hasher> {
    /// A BaseTree contains a single Store.
    Oct(OctMerkleTree<H>),

    /// A SubTree contains a list of BaseTrees.
    OctSub(OctSubMerkleTree<H>),

    /// A TopTree contains a list of SubTrees.
    OctTop(OctTopMerkleTree<H>),

    /// A BaseTree contains a single Store.
    OctLC(OctLCMerkleTree<H>),

    /// A SubTree contains a list of BaseTrees.
    OctLCSub(OctLCSubMerkleTree<H>),

    /// A TopTree contains a list of SubTrees.
    OctLCTop(OctLCTopMerkleTree<H>),
}

impl<H: Hasher> OctTreeData<H> {
    pub fn octtree(&self) -> Option<&OctMerkleTree<H>> {
        match self {
            OctTreeData::Oct(s) => Some(s),
            _ => None,
        }
    }

    pub fn octsubtree(&self) -> Option<&OctSubMerkleTree<H>> {
        match self {
            OctTreeData::OctSub(s) => Some(s),
            _ => None,
        }
    }

    pub fn octtoptree(&self) -> Option<&OctTopMerkleTree<H>> {
        match self {
            OctTreeData::OctTop(s) => Some(s),
            _ => None,
        }
    }

    pub fn octlctree(&self) -> Option<&OctLCMerkleTree<H>> {
        match self {
            OctTreeData::OctLC(s) => Some(s),
            _ => None,
        }
    }

    pub fn octlcsubtree(&self) -> Option<&OctLCSubMerkleTree<H>> {
        match self {
            OctTreeData::OctLCSub(s) => Some(s),
            _ => None,
        }
    }

    pub fn octlctoptree(&self) -> Option<&OctLCTopMerkleTree<H>> {
        match self {
            OctTreeData::OctLCTop(s) => Some(s),
            _ => None,
        }
    }

    pub fn gen_proof(&self, challenge: usize) -> Result<MerkleProof<H, typenum::U8>> {
        // TODO: make this properly generic over a tree
        todo!()
        // match sector_size {
        //     SECTOR_SIZE_2_KIB | SECTOR_SIZE_8_MIB | SECTOR_SIZE_512_MIB => {
        //         assert!(t_aux.tree_r_last.octlctree().is_some());
        //         let tree_r_last = t_aux.tree_r_last.octlctree().unwrap();
        //         let tree_r_last_proof = if t_aux.tree_r_last_config_levels == 0 {
        //             tree_r_last.gen_proof(challenge)
        //         } else {
        //             tree_r_last.gen_cached_proof(challenge, t_aux.tree_r_last_config_levels)
        //         }?;

        //         let comm_r_last_proof = tree_r_last_proof;
        //         assert!(comm_r_last_proof.validate(challenge));

        //         comm_r_last_proof
        //     }
        //     SECTOR_SIZE_4_KIB | SECTOR_SIZE_16_MIB | SECTOR_SIZE_1_GIB | SECTOR_SIZE_32_GIB => {
        //         let sub_tree_count = 2;
        //         let base_tree_leafs = base_tree_leafs / sub_tree_count;

        //         assert!(t_aux.tree_r_last.octlcsubtree().is_some());
        //         let tree_r_last = t_aux.tree_r_last.octlcsubtree().unwrap();
        //         let tree_r_last_proof = if t_aux.tree_r_last_config_levels == 0 {
        //             tree_r_last.gen_proof(challenge)
        //         } else {
        //             tree_r_last.gen_cached_proof(challenge, t_aux.tree_r_last_config_levels)
        //         }?;

        //         assert!(tree_r_last_proof.verify());

        //         let comm_r_last_proof = tree_r_last_proof;
        //         assert!(comm_r_last_proof.validate(challenge));

        //         comm_r_last_proof
        //     }
        //     SECTOR_SIZE_32_KIB | SECTOR_SIZE_64_GIB => {
        //         let top_tree_count = 2;
        //         let sub_tree_count = 8;
        //         let base_tree_leafs = base_tree_leafs / (top_tree_count * sub_tree_count);

        //         assert!(t_aux.tree_r_last.octlctoptree().is_some());
        //         let tree_r_last = t_aux.tree_r_last.octlctoptree().unwrap();
        //         let tree_r_last_proof = if t_aux.tree_r_last_config_levels == 0 {
        //             tree_r_last.gen_proof(challenge)
        //         } else {
        //             tree_r_last.gen_cached_proof(challenge, t_aux.tree_r_last_config_levels)
        //         }?;

        //         assert!(tree_r_last_proof.verify());
        //         let comm_r_last_proof = tree_r_last_proof;

        //         assert!(comm_r_last_proof.validate(challenge));

        //         comm_r_last_proof
        //     }
        //     _ => panic!("Unsupported sector size"),
        // }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PathElement<H: Hasher> {
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    hashes: Vec<H::Domain>,
    index: usize,
}

/// Representation of a merkle proof.
#[derive(Clone, Serialize, Deserialize)]
pub struct MerkleProof<
    H: Hasher,
    BaseArity: Unsigned,
    SubTreeArity: Unsigned = U0,
    TopTreeArity: Unsigned = U0,
> {
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    data: ProofData<H, BaseArity, SubTreeArity, TopTreeArity>,
}

impl<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned, TopTreeArity: Unsigned> std::fmt::Debug
    for MerkleProof<H, BaseArity, SubTreeArity, TopTreeArity>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MerkleProof")
            .field("data", &self.data)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum ProofData<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned, TopTreeArity: Unsigned> {
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    Single(SingleProof<H, BaseArity>),
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    Sub(SubProof<H, BaseArity, SubTreeArity>),
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    Top(TopProof<H, BaseArity, SubTreeArity, TopTreeArity>),
}

impl<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned, TopTreeArity: Unsigned> std::fmt::Debug
    for ProofData<H, BaseArity, SubTreeArity, TopTreeArity>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofData::Single(ref proof) => write!(f, "ProofData::Single({:?})", proof),
            ProofData::Sub(ref proof) => write!(f, "ProofData::Sub({:?})", proof),
            ProofData::Top(ref proof) => write!(f, "ProofData::Top({:?})", proof),
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct SingleProof<H: Hasher, Arity: Unsigned> {
    /// Root of the merkle tree.
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    root: H::Domain,
    /// The original leaf data for this prof.
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    leaf: H::Domain,
    /// The path from leaf to root.
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    path: Vec<PathElement<H>>,
    #[serde(skip)]
    h: PhantomData<H>,
    #[serde(skip)]
    a: PhantomData<Arity>,
}

impl<H: Hasher, Arity: Unsigned> std::fmt::Debug for SingleProof<H, Arity> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleProof")
            .field("root", &self.root)
            .field("leaf", &self.leaf)
            .field("path", &self.path)
            .field("Hasher", &H::name())
            .field("Arity", &Arity::to_usize())
            .finish()
    }
}

impl<H: Hasher, Arity: Unsigned> SingleProof<H, Arity> {
    pub fn new(root: H::Domain, leaf: H::Domain, path: Vec<PathElement<H>>) -> Self {
        SingleProof {
            root,
            leaf,
            path,
            h: Default::default(),
            a: Default::default(),
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct SubProof<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned> {
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    base_proof: SingleProof<H, BaseArity>,
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    sub_proof: SingleProof<H, SubTreeArity>,
    #[serde(skip)]
    h: PhantomData<H>,
    #[serde(skip)]
    b: PhantomData<SubTreeArity>,
}

impl<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned> std::fmt::Debug
    for SubProof<H, BaseArity, SubTreeArity>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubProof")
            .field("base_proof", &self.base_proof)
            .field("sub_proof", &self.sub_proof)
            .finish()
    }
}

impl<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned> SubProof<H, BaseArity, SubTreeArity> {
    pub fn new(
        base_proof: SingleProof<H, BaseArity>,
        sub_proof: SingleProof<H, SubTreeArity>,
    ) -> Self {
        Self {
            base_proof,
            sub_proof,
            h: Default::default(),
            b: Default::default(),
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct TopProof<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned, TopTreeArity: Unsigned> {
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    base_proof: SingleProof<H, BaseArity>,
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    sub_proof: SingleProof<H, SubTreeArity>,
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    top_proof: SingleProof<H, TopTreeArity>,
    #[serde(skip)]
    h: PhantomData<H>,
    #[serde(skip)]
    c: PhantomData<TopTreeArity>,
}

impl<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned, TopTreeArity: Unsigned> std::fmt::Debug
    for TopProof<H, BaseArity, SubTreeArity, TopTreeArity>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopProof")
            .field("base_proof", &self.base_proof)
            .field("sub_proof", &self.sub_proof)
            .field("top_proof", &self.top_proof)
            .finish()
    }
}

impl<H: Hasher, BaseArity: Unsigned, SubTreeArity: Unsigned, TopTreeArity: Unsigned>
    TopProof<H, BaseArity, SubTreeArity, TopTreeArity>
{
    pub fn new(
        base_proof: SingleProof<H, BaseArity>,
        sub_proof: SingleProof<H, SubTreeArity>,
        top_proof: SingleProof<H, TopTreeArity>,
    ) -> Self {
        Self {
            base_proof,
            sub_proof,
            top_proof,
            h: Default::default(),
            c: Default::default(),
        }
    }
}

impl<
        H: Hasher,
        BaseArity: typenum::Unsigned,
        SubTreeArity: typenum::Unsigned,
        TopTreeArity: typenum::Unsigned,
    > MerkleProof<H, BaseArity, SubTreeArity, TopTreeArity>
{
    pub fn new(n: usize) -> Self {
        let root = Default::default();
        let leaf = Default::default();
        let path_elem = PathElement {
            hashes: vec![Default::default()],
            index: 0,
        };
        let path = vec![path_elem; n];
        MerkleProof {
            data: ProofData::Single(SingleProof::new(root, leaf, path)),
        }
    }
}

/// Converts a merkle_light proof to a SingleProof
fn proof_to_single<H: Hasher, Arity: Unsigned, TargetArity: Unsigned>(
    proof: &proof::Proof<H::Domain, Arity>,
    lemma_start_index: usize,
    sub_root: Option<H::Domain>,
) -> SingleProof<H, TargetArity> {
    let root = proof.root();
    let leaf = if sub_root.is_some() {
        sub_root.unwrap()
    } else {
        proof.item()
    };
    let path = extract_path::<H, TargetArity>(proof.lemma(), proof.path(), lemma_start_index);

    SingleProof::new(root, leaf, path)
}

/// 'lemma_start_index' is required because sub/top proofs start at
/// index 0 and base proofs start at index 1 (skipping the leaf at the
/// front)
fn extract_path<H: Hasher, Arity: Unsigned>(
    lemma: &Vec<H::Domain>,
    path: &Vec<usize>,
    lemma_start_index: usize,
) -> Vec<PathElement<H>> {
    lemma[lemma_start_index..lemma.len() - 1]
        .chunks(Arity::to_usize() - 1)
        .zip(path.iter())
        .map(|(hashes, index)| PathElement {
            hashes: hashes.to_vec(),
            index: *index,
        })
        .collect::<Vec<_>>()
}

impl<H: Hasher, Arity: 'static + PoseidonArity> MerkleProofTrait for SingleProof<H, Arity> {
    type Hasher = H;
    type Arity = Arity;
    type SubTreeArity = typenum::U0;
    type TopTreeArity = typenum::U0;

    fn new_from_proof(
        p: &proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
    ) -> Result<Self> {
        Ok(proof_to_single(p, 1, None))
    }

    fn verify(&self) -> bool {
        let mut a = H::Function::default();
        let expected_root = (0..self.path.len()).fold(self.leaf, |h, i| {
            a.reset();

            let index = self.path[i].index;
            let mut nodes = self.path[i].hashes.clone();
            nodes.insert(index, h);

            a.multi_node(&nodes, i)
        });

        self.root() == &expected_root
    }

    fn leaf(&self) -> H::Domain {
        self.leaf
    }

    fn root(&self) -> &H::Domain {
        &self.root
    }

    fn len(&self) -> usize {
        self.path.len() * (Arity::to_usize() - 1) + 2
    }

    fn path(&self) -> Vec<(Vec<H::Domain>, usize)> {
        self.path
            .iter()
            .map(|x| (x.hashes.clone(), x.index))
            .collect::<Vec<_>>()
    }

    fn path_index(&self) -> usize {
        self.path()
            .iter()
            .rev()
            .fold(0, |acc, (_, index)| (acc * Self::Arity::to_usize()) + index)
    }

    fn proves_challenge(&self, challenge: usize) -> bool {
        self.path_index() == challenge
    }
}

impl<H: Hasher, BaseArity: 'static + PoseidonArity, SubTreeArity: 'static + PoseidonArity>
    MerkleProofTrait for SubProof<H, BaseArity, SubTreeArity>
{
    type Hasher = H;
    type Arity = BaseArity;
    type SubTreeArity = SubTreeArity;
    type TopTreeArity = typenum::U0;

    fn new_from_proof(
        p: &proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
    ) -> Result<Self> {
        ensure!(
            p.sub_layer_nodes() == Self::SubTreeArity::to_usize(),
            "sub arity mismatch"
        );
        ensure!(
            p.sub_tree_proof.is_some(),
            "Cannot generate sub proof without a base-proof"
        );
        let base_p = p.sub_tree_proof.as_ref().unwrap();

        // Generate SubProof
        let base_proof = proof_to_single(base_p, 1, None);
        let sub_proof = proof_to_single(p, 0, Some(base_p.root()));

        Ok(SubProof::new(base_proof, sub_proof))
    }

    fn verify(&self) -> bool {
        let base_proof_verifies = self.base_proof.verify();

        if !base_proof_verifies {
            dbg!(base_proof_verifies);
            return false;
        }

        let mut a = H::Function::default();
        let expected_root = (0..self.sub_proof.path.len()).fold(self.base_proof.root, |h, i| {
            a.reset();

            let index = self.sub_proof.path[i].index;
            let mut nodes = self.sub_proof.path[i].hashes.clone();
            nodes.insert(index, h);

            a.multi_node(&nodes, i)
        });
        dbg!(
            self.base_proof.path().len(),
            self.sub_proof.path().len(),
            self.path().len(),
            self.root(),
            &expected_root
        );

        self.root() == &expected_root
    }

    fn leaf(&self) -> H::Domain {
        self.base_proof.leaf()
    }

    fn root(&self) -> &H::Domain {
        self.sub_proof.root()
    }

    fn len(&self) -> usize {
        Self::SubTreeArity::to_usize()
    }

    fn path(&self) -> Vec<(Vec<H::Domain>, usize)> {
        let mut path = self.base_proof.path();
        path.extend_from_slice(&self.sub_proof.path());
        path
    }

    fn path_index(&self) -> usize {
        let mut base_proof_leaves = 1;
        for i in 0..self.base_proof.path().len() {
            base_proof_leaves *= BaseArity::to_usize()
        }

        let sub_proof_index = self
            .sub_proof
            .path()
            .iter()
            .rev()
            .fold(0, |acc, (_, index)| {
                (acc * Self::SubTreeArity::to_usize()) + index
            });
        (sub_proof_index * base_proof_leaves) + self.base_proof.path_index()
    }

    fn proves_challenge(&self, challenge: usize) -> bool {
        let sub_path_index = ((challenge / Self::Arity::to_usize()) * Self::Arity::to_usize())
            + (challenge % Self::Arity::to_usize());

        sub_path_index == challenge
    }
}

impl<
        H: Hasher,
        BaseArity: 'static + PoseidonArity,
        SubTreeArity: 'static + PoseidonArity,
        TopTreeArity: 'static + PoseidonArity,
    > MerkleProofTrait for TopProof<H, BaseArity, SubTreeArity, TopTreeArity>
{
    type Hasher = H;
    type Arity = BaseArity;
    type SubTreeArity = SubTreeArity;
    type TopTreeArity = TopTreeArity;

    fn new_from_proof(
        p: &proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
    ) -> Result<Self> {
        ensure!(
            p.top_layer_nodes() == Self::TopTreeArity::to_usize(),
            "top arity mismatch"
        );
        ensure!(
            p.sub_layer_nodes() == Self::SubTreeArity::to_usize(),
            "sub arity mismatch"
        );

        ensure!(
            p.sub_tree_proof.is_some(),
            "Cannot generate top proof without a sub-proof"
        );
        let sub_p = p.sub_tree_proof.as_ref().unwrap();

        ensure!(
            sub_p.sub_tree_proof.is_some(),
            "Cannot generate top proof without a base-proof"
        );
        let base_p = sub_p.sub_tree_proof.as_ref().unwrap();

        let base_proof = proof_to_single::<Self::Hasher, Self::Arity, Self::Arity>(base_p, 1, None);
        let sub_proof = proof_to_single::<Self::Hasher, Self::Arity, Self::SubTreeArity>(
            sub_p,
            0,
            Some(base_p.root()),
        );
        let top_proof = proof_to_single::<Self::Hasher, Self::Arity, Self::TopTreeArity>(
            p,
            0,
            Some(sub_p.root()),
        );

        assert!(base_proof.verify());
        assert!(sub_proof.verify());
        assert!(top_proof.verify());

        let top = TopProof::new(base_proof, sub_proof, top_proof);
        assert!(top.verify());
        Ok(top)
    }

    fn verify(&self) -> bool {
        let sub_proof_verifies = self.sub_proof.verify();
        let base_proof_verifies = self.base_proof.verify();

        if !sub_proof_verifies || !base_proof_verifies {
            dbg!(sub_proof_verifies, base_proof_verifies);
            return false;
        }

        let mut a = H::Function::default();
        let expected_root = (0..self.top_proof.path.len()).fold(self.sub_proof.root, |h, i| {
            a.reset();

            let index = self.top_proof.path[i].index;
            let mut nodes = self.top_proof.path[i].hashes.clone();
            nodes.insert(index, h);

            a.multi_node(&nodes, i)
        });

        dbg!(self.root(), &expected_root);

        self.root() == &expected_root
    }

    fn validate_data(&self, data: H::Domain) -> bool {
        if !self.verify() {
            return false;
        }
        self.leaf() == data
    }

    fn leaf(&self) -> H::Domain {
        self.base_proof.leaf()
    }

    fn root(&self) -> &H::Domain {
        self.top_proof.root()
    }

    fn len(&self) -> usize {
        TopTreeArity::to_usize()
    }

    fn path(&self) -> Vec<(Vec<H::Domain>, usize)> {
        let mut path = self.base_proof.path();
        path.extend_from_slice(&self.sub_proof.path());
        path.extend_from_slice(&self.top_proof.path());
        path
    }

    fn path_index(&self) -> usize {
        let mut base_proof_leaves = 1;
        for i in 0..self.base_proof.path().len() {
            base_proof_leaves *= BaseArity::to_usize()
        }

        let mut sub_proof_leaves = base_proof_leaves * SubTreeArity::to_usize();

        let sub_proof_index = self
            .sub_proof
            .path()
            .iter()
            .rev()
            .fold(0, |acc, (_, index)| {
                (acc * Self::TopTreeArity::to_usize()) + index
            });

        let top_proof_index = self
            .top_proof
            .path()
            .iter()
            .rev()
            .fold(0, |acc, (_, index)| {
                (acc * Self::TopTreeArity::to_usize()) + index
            });

        (sub_proof_index * base_proof_leaves)
            + (top_proof_index * sub_proof_leaves)
            + self.base_proof.path_index()
    }

    fn proves_challenge(&self, challenge: usize) -> bool {
        let sub_path_index = ((challenge / Self::Arity::to_usize()) * Self::Arity::to_usize())
            + (challenge % Self::Arity::to_usize());

        sub_path_index == challenge
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncludedNode<H: Hasher> {
    value: H::Domain,
    _h: PhantomData<H>,
}

impl<H: Hasher> IncludedNode<H> {
    pub fn new(value: H::Domain) -> Self {
        IncludedNode {
            value,
            _h: PhantomData,
        }
    }

    pub fn into_fr(self) -> Fr {
        self.value.into()
    }
}

impl<H: Hasher> std::ops::Deref for IncludedNode<H> {
    type Target = H::Domain;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

pub fn split_config(config: Option<StoreConfig>, count: usize) -> Result<Vec<Option<StoreConfig>>> {
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

/// Construct a new merkle tree.
pub fn create_merkle_tree<Tree: MerkleTreeTrait>(
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

#[cfg(test)]
mod tests {
    use super::*;

    use rand;
    use std::io::Write;

    use crate::drgraph::{new_seed, BucketGraph, Graph, BASE_DEGREE};
    use crate::hasher::{Blake2sHasher, PedersenHasher, PoseidonHasher, Sha256Hasher};
    use crate::merkle::MerkleProofTrait;

    fn merklepath<H: Hasher, BaseTreeArity: 'static + PoseidonArity>() {
        let leafs = 64;
        let g = BucketGraph::<H>::new(leafs, BASE_DEGREE, 0, new_seed()).unwrap();
        let mut rng = rand::thread_rng();
        let node_size = 32;
        let mut data = Vec::new();
        for _ in 0..leafs {
            let elt: H::Domain = H::Domain::random(&mut rng);
            let bytes = H::Domain::into_bytes(&elt);
            data.write(&bytes).unwrap();
        }

        let tree = g
            .merkle_tree::<MerkleTreeWrapper<H, DiskStore<H::Domain>, BaseTreeArity, typenum::U0, typenum::U0>>(
                None,
                data.as_slice(),
            )
            .unwrap();
        for i in 0..leafs {
            let proof = tree.gen_proof(i).unwrap();

            assert!(proof.verify(), "failed to validate");

            assert!(proof.validate(i), "failed to validate valid merkle path");
            let data_slice = &data[i * node_size..(i + 1) * node_size].to_vec();
            assert!(
                proof.validate_data(H::Domain::try_from_bytes(data_slice).unwrap()),
                "failed to validate valid data"
            );
        }
    }

    fn merklepath_sub<
        H: Hasher,
        BaseTreeArity: 'static + PoseidonArity,
        SubTreeArity: 'static + PoseidonArity,
    >() {
        let leafs = 64;
        let g = BucketGraph::<H>::new(leafs, BASE_DEGREE, 0, new_seed()).unwrap();
        let mut rng = rand::thread_rng();
        let node_size = 32;
        let mut data = Vec::new();
        for _ in 0..leafs {
            let elt: H::Domain = H::Domain::random(&mut rng);
            data.push(elt);
        }

        let mut trees = Vec::with_capacity(SubTreeArity::to_usize());
        for i in 0..SubTreeArity::to_usize() {
            trees.push(
                merkletree::merkle::MerkleTree::<
                    H::Domain,
                    H::Function,
                    DiskStore<_>,
                    BaseTreeArity,
                >::try_from_iter(data.clone().into_iter().map(Ok))
                .expect("failed to build tree"),
            );
        }

        let tree = merkletree::merkle::MerkleTree::<
            H::Domain,
            H::Function,
            DiskStore<_>,
            BaseTreeArity,
            SubTreeArity,
        >::from_trees(trees)
        .expect("Failed to build 2 layer tree");

        for i in 0..(leafs * SubTreeArity::to_usize()) {
            let proof = SubProof::<H, BaseTreeArity, SubTreeArity>::new_from_proof(
                &tree.gen_proof(i).unwrap(),
            )
            .expect("failed to build sub-proof");
            assert_eq!(proof.base_proof.root, proof.sub_proof.leaf);
            assert!(proof.verify(), "failed to validate");

            assert!(proof.validate(i), "failed to validate valid merkle path");
            let data: H::Domain = tree.read_at(i).expect("failed to read data from tree");
            assert!(
                proof.validate_data(H::Domain::try_from_bytes(&data.into_bytes()).unwrap()),
                "failed to validate valid data"
            );
        }
    }

    fn merklepath_top<
        H: Hasher,
        BaseTreeArity: 'static + PoseidonArity,
        SubTreeArity: 'static + PoseidonArity,
        TopTreeArity: 'static + PoseidonArity,
    >() {
        let leafs = 64;
        let g = BucketGraph::<H>::new(leafs, BASE_DEGREE, 0, new_seed()).unwrap();
        let mut rng = rand::thread_rng();
        let node_size = 32;
        let mut data = Vec::new();
        for _ in 0..leafs {
            let elt: H::Domain = H::Domain::random(&mut rng);
            data.push(elt);
        }

        let mut sub_trees = Vec::with_capacity(TopTreeArity::to_usize());
        for i in 0..TopTreeArity::to_usize() {
            let mut trees = Vec::with_capacity(SubTreeArity::to_usize());
            for i in 0..SubTreeArity::to_usize() {
                trees.push(
                    merkletree::merkle::MerkleTree::<
                        H::Domain,
                        H::Function,
                        DiskStore<_>,
                        BaseTreeArity,
                    >::try_from_iter(data.clone().into_iter().map(Ok))
                    .expect("failed to build tree"),
                );
            }

            sub_trees.push(
                merkletree::merkle::MerkleTree::<
                    H::Domain,
                    H::Function,
                    DiskStore<_>,
                    BaseTreeArity,
                    SubTreeArity,
                >::from_trees(trees)
                .expect("Failed to build 2 layer tree"),
            );
        }

        let tree = merkletree::merkle::MerkleTree::<
            H::Domain,
            H::Function,
            DiskStore<_>,
            BaseTreeArity,
            SubTreeArity,
            TopTreeArity,
        >::from_sub_trees(sub_trees)
        .expect("Failed to build 3 layer tree");

        for i in 0..(leafs * SubTreeArity::to_usize() * TopTreeArity::to_usize()) {
            let proof = TopProof::<H, BaseTreeArity, SubTreeArity, TopTreeArity>::new_from_proof(
                &tree.gen_proof(i).unwrap(),
            )
            .expect("failed to build top-proof");

            assert!(proof.verify(), "failed to validate");

            assert!(proof.validate(i), "failed to validate valid merkle path");
            let data: H::Domain = tree.read_at(i).expect("failed to read data from tree");
            assert!(
                proof.validate_data(H::Domain::try_from_bytes(&data.into_bytes()).unwrap()),
                "failed to validate valid data"
            );
        }
    }

    #[test]
    fn merklepath_pedersen_binary() {
        merklepath::<PedersenHasher, typenum::U2>();
    }

    #[test]
    fn merklepath_pedersen_sub_binary_binary() {
        merklepath_sub::<PedersenHasher, typenum::U2, typenum::U2>();
    }

    #[test]
    fn merklepath_poseidon_sub_oct_binary() {
        merklepath_sub::<PoseidonHasher, typenum::U8, typenum::U2>();
    }

    #[test]
    fn merklepath_poseidon_sub_oct_quad() {
        merklepath_sub::<PoseidonHasher, typenum::U8, typenum::U4>();
    }

    #[test]
    fn merklepath_pedersen_top_binary_binary_binary() {
        merklepath_top::<PedersenHasher, typenum::U2, typenum::U2, typenum::U2>();
    }

    #[test]
    fn merklepath_poseidon_top_oct_quad_binary() {
        merklepath_top::<PoseidonHasher, typenum::U8, typenum::U4, typenum::U2>();
    }

    #[test]
    fn merklepath_sha256_binary() {
        merklepath::<Sha256Hasher, typenum::U2>();
    }

    #[test]
    fn merklepath_sha256_sub_binary_quad() {
        merklepath_sub::<Sha256Hasher, typenum::U2, typenum::U4>();
    }

    #[test]
    fn merklepath_sha256_top_binary_quad_binary() {
        merklepath_top::<Sha256Hasher, typenum::U2, typenum::U4, typenum::U2>();
    }

    #[test]
    fn merklepath_blake2s_binary() {
        merklepath::<Blake2sHasher, typenum::U2>();
    }

    #[test]
    fn merklepath_poseidon_binary() {
        merklepath::<PoseidonHasher, typenum::U2>();
    }

    #[test]
    fn merklepath_poseidon_quad() {
        merklepath::<PoseidonHasher, typenum::U4>();
    }

    #[test]
    fn merklepath_pedersen_quad() {
        merklepath::<PedersenHasher, typenum::U4>();
    }

    #[test]
    fn merklepath_sha256_quad() {
        merklepath::<Sha256Hasher, typenum::U4>();
    }

    #[test]
    fn merklepath_blake2s_quad() {
        merklepath::<Blake2sHasher, typenum::U4>();
    }

    #[test]
    fn merklepath_poseidon_oct() {
        merklepath::<PoseidonHasher, typenum::U8>();
    }

    #[test]
    fn merklepath_pedersen_oct() {
        merklepath::<PedersenHasher, typenum::U8>();
    }
}
