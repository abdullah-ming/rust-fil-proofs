use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use anyhow::Context;
use generic_array::typenum::{self, Unsigned};
use log::trace;
use merkletree::merkle::get_merkle_tree_leafs;
use merkletree::store::{DiskStore, Store, StoreConfig};
use serde::{Deserialize, Serialize};

use super::{
    column::Column, column_proof::ColumnProof, graph::StackedBucketGraph, EncodingProof,
    LabelingProof, LayerChallenges,
};

use crate::drgraph::Graph;
use crate::error::Result;
use crate::fr32::bytes_into_fr_repr_safe;
use crate::hasher::{Domain, Hasher};
use crate::merkle::*;
use crate::parameter_cache::ParameterSetMetadata;
use crate::util::data_at_node;

pub const BINARY_ARITY: usize = 2;
pub const QUAD_ARITY: usize = 4;
pub const OCT_ARITY: usize = 8;

#[derive(Debug, Clone)]
pub struct SetupParams {
    // Number of nodes
    pub nodes: usize,

    // Base degree of DRG
    pub degree: usize,

    pub expansion_degree: usize,

    // Random seed
    pub seed: [u8; 28],

    pub layer_challenges: LayerChallenges,
}

#[derive(Debug, Clone)]
pub struct PublicParams<H>
where
    H: 'static + Hasher,
{
    pub graph: StackedBucketGraph<H>,
    pub layer_challenges: LayerChallenges,
    _h: PhantomData<H>,
}

impl<H> PublicParams<H>
where
    H: Hasher,
{
    pub fn new(graph: StackedBucketGraph<H>, layer_challenges: LayerChallenges) -> Self {
        PublicParams {
            graph,
            layer_challenges,
            _h: PhantomData,
        }
    }
}

impl<H> ParameterSetMetadata for PublicParams<H>
where
    H: Hasher,
{
    fn identifier(&self) -> String {
        format!(
            "layered_drgporep::PublicParams{{ graph: {}, challenges: {:?} }}",
            self.graph.identifier(),
            self.layer_challenges,
        )
    }

    fn sector_size(&self) -> u64 {
        self.graph.sector_size()
    }
}

impl<'a, H> From<&'a PublicParams<H>> for PublicParams<H>
where
    H: Hasher,
{
    fn from(other: &PublicParams<H>) -> PublicParams<H> {
        PublicParams::new(other.graph.clone(), other.layer_challenges.clone())
    }
}

#[derive(Debug, Clone)]
pub struct PublicInputs<T: Domain, S: Domain> {
    pub replica_id: T,
    pub seed: [u8; 32],
    pub tau: Option<Tau<T, S>>,
    /// Partition index
    pub k: Option<usize>,
}

impl<T: Domain, S: Domain> PublicInputs<T, S> {
    pub fn challenges(
        &self,
        layer_challenges: &LayerChallenges,
        leaves: usize,
        partition_k: Option<usize>,
    ) -> Vec<usize> {
        let k = partition_k.unwrap_or(0);

        layer_challenges.derive::<T>(leaves, &self.replica_id, &self.seed, k as u8)
    }
}

#[derive(Debug)]
pub struct PrivateInputs<Tree: MerkleTreeTrait, G: Hasher> {
    pub p_aux: PersistentAux<<Tree::Hasher as Hasher>::Domain>,
    pub t_aux: TemporaryAuxCache<Tree, G>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Proof<Tree: MerkleTreeTrait, G: Hasher> {
    #[serde(bound(
        serialize = "MerkleProof<G, typenum::U2>: Serialize",
        deserialize = "MerkleProof<G, typenum::U2>: Deserialize<'de>"
    ))]
    pub comm_d_proofs: MerkleProof<G, typenum::U2>,
    #[serde(bound(
        serialize = "MerkleProof<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>: Serialize",
        deserialize = "MerkleProof<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>: Deserialize<'de>"
    ))]
    pub comm_r_last_proof:
        MerkleProof<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>,
    #[serde(bound(
        serialize = "ReplicaColumnProof<Tree::Hasher>: Serialize",
        deserialize = "ReplicaColumnProof<Tree::Hasher>: Deserialize<'de>"
    ))]
    pub replica_column_proofs: ReplicaColumnProof<Tree::Hasher>,
    #[serde(bound(
        serialize = "LabelingProof<Tree::Hasher>: Serialize",
        deserialize = "LabelingProof<Tree::Hasher>: Deserialize<'de>"
    ))]
    /// Indexed by layer in 1..layers.
    pub labeling_proofs: HashMap<usize, LabelingProof<Tree::Hasher>>,
    #[serde(bound(
        serialize = "EncodingProof<Tree::Hasher>: Serialize",
        deserialize = "EncodingProof<Tree::Hasher>: Deserialize<'de>"
    ))]
    pub encoding_proof: EncodingProof<Tree::Hasher>,
}

impl<Tree: MerkleTreeTrait, G: Hasher> Clone for Proof<Tree, G> {
    fn clone(&self) -> Self {
        Self {
            comm_d_proofs: self.comm_d_proofs.clone(),
            comm_r_last_proof: self.comm_r_last_proof.clone(),
            replica_column_proofs: self.replica_column_proofs.clone(),
            labeling_proofs: self.labeling_proofs.clone(),
            encoding_proof: self.encoding_proof.clone(),
        }
    }
}

impl<Tree: MerkleTreeTrait, G: Hasher> Proof<Tree, G> {
    pub fn comm_r_last(&self) -> &<Tree::Hasher as Hasher>::Domain {
        self.comm_r_last_proof.root()
    }

    pub fn comm_c(&self) -> &<Tree::Hasher as Hasher>::Domain {
        self.replica_column_proofs.c_x.root()
    }

    /// Verify the full proof.
    pub fn verify(
        &self,
        pub_params: &PublicParams<Tree::Hasher>,
        pub_inputs: &PublicInputs<<Tree::Hasher as Hasher>::Domain, <G as Hasher>::Domain>,
        challenge: usize,
        graph: &StackedBucketGraph<Tree::Hasher>,
    ) -> bool {
        let replica_id = &pub_inputs.replica_id;

        check!(challenge < graph.size());
        check!(pub_inputs.tau.is_some());

        // Verify initial data layer
        trace!("verify initial data layer");

        check!(self.comm_d_proofs.proves_challenge(challenge));

        if let Some(ref tau) = pub_inputs.tau {
            check_eq!(self.comm_d_proofs.root(), &tau.comm_d);
        } else {
            return false;
        }

        // Verify replica column openings
        trace!("verify replica column openings");
        let mut parents = vec![0; graph.degree()];
        graph.parents(challenge, &mut parents).unwrap(); // FIXME: error handling
        check!(self.replica_column_proofs.verify(challenge, &parents));

        check!(self.verify_final_replica_layer(challenge));

        check!(self.verify_labels(replica_id, &pub_params.layer_challenges));

        trace!("verify encoding");
        let sector_size =
            (graph.size() * std::mem::size_of::<<Tree::Hasher as Hasher>::Domain>()) as u64;
        todo!();
        // match sector_size {
        //     SECTOR_SIZE_2_KIB | SECTOR_SIZE_8_MIB | SECTOR_SIZE_512_MIB => {
        //         check!(self.encoding_proof.verify::<G>(
        //             replica_id,
        //             &self.comm_r_last_proof.leaf(),
        //             &self.comm_d_proofs.leaf()
        //         ));
        //     }
        //     SECTOR_SIZE_4_KIB | SECTOR_SIZE_16_MIB | SECTOR_SIZE_1_GIB | SECTOR_SIZE_32_GIB => {
        //         assert!(self.comm_r_last_proof.sub_tree_proof.is_some());
        //         check!(self.encoding_proof.verify::<G>(
        //             replica_id,
        //             &self
        //                 .comm_r_last_proof
        //                 .sub_tree_proof
        //                 .as_ref()
        //                 .unwrap()
        //                 .leaf(),
        //             &self.comm_d_proofs.leaf()
        //         ));
        //     }
        //     SECTOR_SIZE_32_KIB | SECTOR_SIZE_64_GIB => {
        //         assert!(self.comm_r_last_proof.sub_tree_proof.is_some());
        //         assert!(self
        //             .comm_r_last_proof
        //             .sub_tree_proof
        //             .as_ref()
        //             .unwrap()
        //             .sub_tree_proof
        //             .is_some());
        //         check!(self.encoding_proof.verify::<G>(
        //             replica_id,
        //             &self
        //                 .comm_r_last_proof
        //                 .sub_tree_proof
        //                 .as_ref()
        //                 .unwrap()
        //                 .sub_tree_proof
        //                 .as_ref()
        //                 .unwrap()
        //                 .leaf(),
        //             &self.comm_d_proofs.leaf()
        //         ));
        //     }
        //     _ => panic!("Unsupported sector size"),
        // }

        true
    }

    /// Verify all labels.
    fn verify_labels(
        &self,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        layer_challenges: &LayerChallenges,
    ) -> bool {
        // Verify Labels Layer 1..layers
        for layer in 1..=layer_challenges.layers() {
            trace!("verify labeling (layer: {})", layer,);

            check!(self.labeling_proofs.contains_key(&layer));
            let labeling_proof = &self.labeling_proofs.get(&layer).unwrap();
            let labeled_node = self
                .replica_column_proofs
                .c_x
                .get_node_at_layer(layer)
                .unwrap(); // FIXME: error handling
            check!(labeling_proof.verify(replica_id, labeled_node));
        }

        true
    }

    /// Verify final replica layer openings
    fn verify_final_replica_layer(&self, challenge: usize) -> bool {
        trace!("verify final replica layer openings");
        check!(self.comm_r_last_proof.proves_challenge(challenge));

        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaColumnProof<H: Hasher> {
    #[serde(bound(
        serialize = "ColumnProof<H>: Serialize",
        deserialize = "ColumnProof<H>: Deserialize<'de>"
    ))]
    pub c_x: ColumnProof<H>,
    #[serde(bound(
        serialize = "ColumnProof<H>: Serialize",
        deserialize = "ColumnProof<H>: Deserialize<'de>"
    ))]
    pub drg_parents: Vec<ColumnProof<H>>,
    #[serde(bound(
        serialize = "ColumnProof<H>: Serialize",
        deserialize = "ColumnProof<H>: Deserialize<'de>"
    ))]
    pub exp_parents: Vec<ColumnProof<H>>,
}

impl<H: Hasher> ReplicaColumnProof<H> {
    pub fn verify(&self, challenge: usize, parents: &[u32]) -> bool {
        let expected_comm_c = self.c_x.root();

        trace!("  verify c_x");
        check!(self.c_x.verify(challenge as u32, &expected_comm_c));

        trace!("  verify drg_parents");
        for (proof, parent) in self.drg_parents.iter().zip(parents.iter()) {
            check!(proof.verify(*parent, &expected_comm_c));
        }

        trace!("  verify exp_parents");
        for (proof, parent) in self
            .exp_parents
            .iter()
            .zip(parents.iter().skip(self.drg_parents.len()))
        {
            check!(proof.verify(*parent, &expected_comm_c));
        }

        true
    }
}

impl<Tree: MerkleTreeTrait, G: Hasher> Proof<Tree, G> {
    pub fn serialize(&self) -> Vec<u8> {
        unimplemented!();
    }
}

pub type TransformedLayers<H, G> = (
    Tau<<H as Hasher>::Domain, <G as Hasher>::Domain>,
    PersistentAux<<H as Hasher>::Domain>,
    TemporaryAux<H, G>,
);

/// Tau for a single parition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tau<D: Domain, E: Domain> {
    pub comm_d: E,
    pub comm_r: D,
}

/// Stored along side the sector on disk.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PersistentAux<D> {
    pub comm_c: D,
    pub comm_r_last: D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporaryAux<H: Hasher, G: Hasher> {
    /// The encoded nodes for 1..layers.
    pub labels: Labels<H>,
    pub tree_d_config: StoreConfig,
    pub tree_r_last_config: StoreConfig,
    pub tree_c_config: StoreConfig,
    pub _g: PhantomData<G>,
}

impl<H: Hasher, G: Hasher> TemporaryAux<H, G> {
    pub fn set_cache_path<P: AsRef<Path>>(&mut self, cache_path: P) {
        let cp = cache_path.as_ref().to_path_buf();
        for label in self.labels.labels.iter_mut() {
            label.path = cp.clone();
        }
        self.tree_d_config.path = cp.clone();
        self.tree_r_last_config.path = cp.clone();
        self.tree_c_config.path = cp;
    }

    pub fn labels_for_layer(&self, layer: usize) -> Result<DiskStore<H::Domain>> {
        self.labels.labels_for_layer(layer)
    }

    pub fn domain_node_at_layer(&self, layer: usize, node_index: u32) -> Result<H::Domain> {
        Ok(self.labels_for_layer(layer)?.read_at(node_index as usize)?)
    }

    pub fn column(&self, column_index: u32) -> Result<Column<H>> {
        self.labels.column(column_index)
    }

    // 'clear_temp' will discard all persisted merkle and layer data
    // that is no longer required.
    pub fn clear_temp(t_aux: TemporaryAux<H, G>) -> Result<()> {
        let cached = |config: &StoreConfig| {
            Path::new(&StoreConfig::data_path(&config.path, &config.id)).exists()
        };

        if cached(&t_aux.tree_d_config) {
            let tree_d_size = t_aux
                .tree_d_config
                .size
                .context("tree_d config has no size")?;
            let tree_d_store: DiskStore<G::Domain> =
                DiskStore::new_from_disk(tree_d_size, BINARY_ARITY, &t_aux.tree_d_config)
                    .context("tree_d")?;
            let tree_d = BinaryMerkleTree::<G>::from_data_store(
                tree_d_store,
                get_merkle_tree_leafs(tree_d_size, BINARY_ARITY),
            )
            .context("tree_d")?;

            tree_d.delete(t_aux.tree_d_config).context("tree_d")?;
            trace!("tree d deleted");
        }

        if cached(&t_aux.tree_c_config) {
            let tree_c_size = t_aux
                .tree_c_config
                .size
                .context("tree_c config has no size")?;
            let tree_c_store: DiskStore<H::Domain> =
                DiskStore::new_from_disk(tree_c_size, OCT_ARITY, &t_aux.tree_c_config)
                    .context("tree_c")?;
            let tree_c: OctMerkleTree<H> = MerkleTree::from_data_store(
                tree_c_store,
                get_merkle_tree_leafs(tree_c_size, OCT_ARITY),
            )
            .context("tree_c")?;
            tree_c.delete(t_aux.tree_c_config).context("tree_c")?;
            trace!("tree c deleted");
        }

        for i in 0..t_aux.labels.labels.len() {
            let cur_config = t_aux.labels.labels[i].clone();
            if cached(&cur_config) {
                DiskStore::<H::Domain>::delete(cur_config)
                    .with_context(|| format!("labels {}", i))?;
                trace!("layer {} deleted", i);
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct TemporaryAuxCache<Tree: MerkleTreeTrait, G: Hasher> {
    /// The encoded nodes for 1..layers.
    pub labels: LabelsCache<Tree>,
    pub tree_d: BinaryMerkleTree<G>,

    // Notably this is a LevelCacheTree instead of a full merkle.
    // pub tree_r_last: OctTreeData<Tree::Hasher>,
    pub tree_r_last: LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>,

    // Store the 'cached_above_base layers' value from the tree_r_last
    // StoreConfig for later use (i.e. proof generation).
    pub tree_r_last_config_levels: usize,

    //pub tree_c: Tree,
    pub tree_c: OctTreeData<Tree::Hasher>,
    pub t_aux: TemporaryAux<Tree::Hasher, G>,
    pub replica_path: PathBuf,
}

impl<Tree: MerkleTreeTrait, G: Hasher> TemporaryAuxCache<Tree, G> {
    pub fn new(t_aux: &TemporaryAux<Tree::Hasher, G>, replica_path: PathBuf) -> Result<Self> {
        let tree_d_size = t_aux.tree_d_config.size.unwrap();
        trace!(
            "Instantiating tree d with size {} and leafs {}",
            tree_d_size,
            get_merkle_tree_leafs(tree_d_size, BINARY_ARITY)
        );
        let tree_d_store: DiskStore<G::Domain> =
            DiskStore::new_from_disk(tree_d_size, BINARY_ARITY, &t_aux.tree_d_config)
                .context("tree_d_store")?;
        let tree_d = BinaryMerkleTree::<G>::from_data_store(
            tree_d_store,
            get_merkle_tree_leafs(tree_d_size, BINARY_ARITY),
        )
        .context("tree_d")?;

        let sector_size =
            get_merkle_tree_leafs(tree_d_size, BINARY_ARITY) * std::mem::size_of::<G::Domain>();

        if Tree::TopTreeArity::to_usize() > 0 {
            // Note: Given the shape of OctTopTree, a tree count
            // of 16 will yield 2 top layer trees, each consisting
            // of 8 sub layer trees.
            let tree_count = 16;

            let configs = {
                let c = split_config(Some(t_aux.tree_c_config.clone()), tree_count)?;
                let mut configs = Vec::with_capacity(tree_count);

                for i in 0..tree_count {
                    assert!(c[i].is_some());
                    configs.push(c[i].as_ref().unwrap().clone());
                }

                configs
            };

            let tree_c_size = t_aux.tree_c_config.size.unwrap();
            let tree_c_leafs = get_merkle_tree_leafs(tree_c_size, OCT_ARITY);
            let tree_c = OctTopMerkleTree::<Tree::Hasher>::from_sub_tree_store_configs(
                tree_c_leafs,
                &configs,
            )?;

            let tree_r_last_config_levels = t_aux.tree_r_last_config.levels;
            let tree_r_last_size = t_aux.tree_r_last_config.size.unwrap();
            let tree_r_last_leafs = get_merkle_tree_leafs(tree_r_last_size, OCT_ARITY);

            let (configs, replica_paths) = {
                let c = split_config(Some(t_aux.tree_r_last_config.clone()), tree_count)?;

                let mut configs = Vec::with_capacity(tree_count);
                let mut replica_paths = Vec::with_capacity(tree_count);

                for i in 0..tree_count {
                    assert!(c[i].is_some());
                    configs.push(c[i].as_ref().unwrap().clone());
                    replica_paths.push(
                        Path::new(
                            format!("{:?}-{}", replica_path, i)
                                .replace("\"", "")
                                .as_str(),
                        )
                        .to_path_buf(),
                    );
                }

                (configs, replica_paths)
            };

            let tree_r_last = LCTree::from_sub_tree_store_configs_and_replicas(
                tree_r_last_leafs,
                &configs,
                &replica_paths,
            )?;

            Ok(TemporaryAuxCache {
                labels: LabelsCache::new(&t_aux.labels).context("labels_cache")?,
                tree_d,
                tree_r_last,
                tree_r_last_config_levels,
                tree_c: OctTreeData::OctTop(tree_c),
                replica_path,
                t_aux: t_aux.clone(),
            })
        } else if <Tree::SubTreeArity as typenum::Unsigned>::to_usize() > 0 {
            let tree_count = if sector_size as u64 == SECTOR_SIZE_32_GIB {
                8
            } else {
                2
            };

            let configs = {
                let c = split_config(Some(t_aux.tree_c_config.clone()), tree_count)?;
                let mut configs = Vec::with_capacity(tree_count);

                for i in 0..tree_count {
                    assert!(c[i].is_some());
                    configs.push(c[i].as_ref().unwrap().clone());
                }

                configs
            };

            let tree_c_size = t_aux.tree_c_config.size.unwrap();
            let tree_c_leafs = get_merkle_tree_leafs(tree_c_size, OCT_ARITY);
            let tree_c =
                OctSubMerkleTree::<Tree::Hasher>::from_store_configs(tree_c_leafs, &configs)?;

            let tree_r_last_config_levels = t_aux.tree_r_last_config.levels;
            let tree_r_last_size = t_aux.tree_r_last_config.size.unwrap();
            let tree_r_last_leafs = get_merkle_tree_leafs(tree_r_last_size, OCT_ARITY);

            let (configs, replica_paths) = {
                let c = split_config(Some(t_aux.tree_r_last_config.clone()), tree_count)?;

                let mut configs = Vec::with_capacity(tree_count);
                let mut replica_paths = Vec::with_capacity(tree_count);

                for i in 0..tree_count {
                    assert!(c[i].is_some());
                    configs.push(c[i].as_ref().unwrap().clone());
                    replica_paths.push(
                        Path::new(
                            format!("{:?}-{}", replica_path, i)
                                .replace("\"", "")
                                .as_str(),
                        )
                        .to_path_buf(),
                    );
                }

                (configs, replica_paths)
            };

            let tree_r_last = LCTree::from_store_configs_and_replicas(
                tree_r_last_leafs,
                &configs,
                &replica_paths,
            )?;

            Ok(TemporaryAuxCache {
                labels: LabelsCache::new(&t_aux.labels).context("labels_cache")?,
                tree_d,
                tree_r_last,
                tree_r_last_config_levels,
                tree_c: OctTreeData::OctSub(tree_c),
                replica_path,
                t_aux: t_aux.clone(),
            })
        } else {
            let tree_c_size = t_aux.tree_c_config.size.unwrap();
            trace!(
                "Instantiating tree c with size {} and leafs {}",
                tree_c_size,
                get_merkle_tree_leafs(tree_c_size, OCT_ARITY)
            );
            let tree_c_store: DiskStore<<Tree::Hasher as Hasher>::Domain> =
                DiskStore::new_from_disk(tree_c_size, OCT_ARITY, &t_aux.tree_c_config)
                    .context("tree_c_store")?;
            let tree_c: OctMerkleTree<Tree::Hasher> = MerkleTree::from_data_store(
                tree_c_store,
                get_merkle_tree_leafs(tree_c_size, OCT_ARITY),
            )
            .context("tree_c")?;

            let tree_r_last_size = t_aux.tree_r_last_config.size.unwrap();
            trace!(
                "Instantiating tree r last with size {} and leafs {}",
                tree_r_last_size,
                get_merkle_tree_leafs(tree_r_last_size, OCT_ARITY)
            );

            let tree_r_last = open_lcmerkle_tree::<
                Tree::Hasher,
                Tree::Arity,
                Tree::SubTreeArity,
                Tree::TopTreeArity,
            >(
                t_aux.tree_r_last_config.clone(),
                get_merkle_tree_leafs(tree_r_last_size, OCT_ARITY),
                &replica_path,
            )?;
            let tree_r_last_config_levels = t_aux.tree_r_last_config.levels;

            Ok(TemporaryAuxCache {
                labels: LabelsCache::new(&t_aux.labels).context("labels_cache")?,
                tree_d,
                tree_r_last,
                tree_r_last_config_levels,
                tree_c: OctTreeData::Oct(tree_c),
                replica_path,
                t_aux: t_aux.clone(),
            })
        }

        /*
        match sector_size as u64 {
            SECTOR_SIZE_2_KIB | SECTOR_SIZE_8_MIB | SECTOR_SIZE_512_MIB => {
                let tree_c_size = t_aux.tree_c_config.size.unwrap();
                trace!(
                    "Instantiating tree c with size {} and leafs {}",
                    tree_c_size,
                    get_merkle_tree_leafs(tree_c_size, OCT_ARITY)
                );
                let tree_c_store: DiskStore<<Tree::Hasher as Hasher>::Domain> =
                    DiskStore::new_from_disk(tree_c_size, OCT_ARITY, &t_aux.tree_c_config)
                        .context("tree_c_store")?;
                let tree_c: OctMerkleTree<Tree::Hasher> = MerkleTree::from_data_store(
                    tree_c_store,
                    get_merkle_tree_leafs(tree_c_size, OCT_ARITY),
                )
                .context("tree_c")?;

                let tree_r_last_size = t_aux.tree_r_last_config.size.unwrap();
                trace!(
                    "Instantiating tree r last with size {} and leafs {}",
                    tree_r_last_size,
                    get_merkle_tree_leafs(tree_r_last_size, OCT_ARITY)
                );

                let tree_r_last: OctLCMerkleTree<Tree::Hasher> =
                    open_lcmerkle_tree::<Tree::Hasher, typenum::U8>(
                        t_aux.tree_r_last_config.clone(),
                        get_merkle_tree_leafs(tree_r_last_size, OCT_ARITY),
                        &replica_path,
                    )?;
                let tree_r_last_config_levels = t_aux.tree_r_last_config.levels;

                Ok(TemporaryAuxCache {
                    labels: LabelsCache::new(&t_aux.labels).context("labels_cache")?,
                    tree_d,
                    tree_r_last: OctTreeData::OctLC(tree_r_last),
                    tree_r_last_config_levels,
                    tree_c: OctTreeData::Oct(tree_c),
                    replica_path,
                    t_aux: t_aux.clone(),
                })
            }
            SECTOR_SIZE_4_KIB | SECTOR_SIZE_16_MIB | SECTOR_SIZE_1_GIB | SECTOR_SIZE_32_GIB => {
                let tree_count = if sector_size as u64 == SECTOR_SIZE_32_GIB {
                    8
                } else {
                    2
                };

                let configs = {
                    let c = split_config(Some(t_aux.tree_c_config.clone()), tree_count)?;
                    let mut configs = Vec::with_capacity(tree_count);

                    for i in 0..tree_count {
                        assert!(c[i].is_some());
                        configs.push(c[i].as_ref().unwrap().clone());
                    }

                    configs
                };

                let tree_c_size = t_aux.tree_c_config.size.unwrap();
                let tree_c_leafs = get_merkle_tree_leafs(tree_c_size, OCT_ARITY);
                let tree_c =
                    OctSubMerkleTree::<Tree::Hasher>::from_store_configs(tree_c_leafs, &configs)?;

                let tree_r_last_config_levels = t_aux.tree_r_last_config.levels;
                let tree_r_last_size = t_aux.tree_r_last_config.size.unwrap();
                let tree_r_last_leafs = get_merkle_tree_leafs(tree_r_last_size, OCT_ARITY);

                let (configs, replica_paths) = {
                    let c = split_config(Some(t_aux.tree_r_last_config.clone()), tree_count)?;

                    let mut configs = Vec::with_capacity(tree_count);
                    let mut replica_paths = Vec::with_capacity(tree_count);

                    for i in 0..tree_count {
                        assert!(c[i].is_some());
                        configs.push(c[i].as_ref().unwrap().clone());
                        replica_paths.push(
                            Path::new(
                                format!("{:?}-{}", replica_path, i)
                                    .replace("\"", "")
                                    .as_str(),
                            )
                            .to_path_buf(),
                        );
                    }

                    (configs, replica_paths)
                };

                let tree_r_last =
                    OctLCSubMerkleTree::<Tree::Hasher>::from_store_configs_and_replicas(
                        tree_r_last_leafs,
                        &configs,
                        &replica_paths,
                    )?;

                Ok(TemporaryAuxCache {
                    labels: LabelsCache::new(&t_aux.labels).context("labels_cache")?,
                    tree_d,
                    tree_r_last: OctTreeData::OctLCSub(tree_r_last),
                    tree_r_last_config_levels,
                    tree_c: OctTreeData::OctSub(tree_c),
                    replica_path,
                    t_aux: t_aux.clone(),
                })
            }
            SECTOR_SIZE_32_KIB | SECTOR_SIZE_64_GIB => {
                // Note: Given the shape of OctTopTree, a tree count
                // of 16 will yield 2 top layer trees, each consisting
                // of 8 sub layer trees.
                let tree_count = 16;

                let configs = {
                    let c = split_config(Some(t_aux.tree_c_config.clone()), tree_count)?;
                    let mut configs = Vec::with_capacity(tree_count);

                    for i in 0..tree_count {
                        assert!(c[i].is_some());
                        configs.push(c[i].as_ref().unwrap().clone());
                    }

                    configs
                };

                let tree_c_size = t_aux.tree_c_config.size.unwrap();
                let tree_c_leafs = get_merkle_tree_leafs(tree_c_size, OCT_ARITY);
                let tree_c = OctTopMerkleTree::<Tree::Hasher>::from_sub_tree_store_configs(
                    tree_c_leafs,
                    &configs,
                )?;

                let tree_r_last_config_levels = t_aux.tree_r_last_config.levels;
                let tree_r_last_size = t_aux.tree_r_last_config.size.unwrap();
                let tree_r_last_leafs = get_merkle_tree_leafs(tree_r_last_size, OCT_ARITY);

                let (configs, replica_paths) = {
                    let c = split_config(Some(t_aux.tree_r_last_config.clone()), tree_count)?;

                    let mut configs = Vec::with_capacity(tree_count);
                    let mut replica_paths = Vec::with_capacity(tree_count);

                    for i in 0..tree_count {
                        assert!(c[i].is_some());
                        configs.push(c[i].as_ref().unwrap().clone());
                        replica_paths.push(
                            Path::new(
                                format!("{:?}-{}", replica_path, i)
                                    .replace("\"", "")
                                    .as_str(),
                            )
                            .to_path_buf(),
                        );
                    }

                    (configs, replica_paths)
                };

                let tree_r_last =
                    OctLCTopMerkleTree::<Tree::Hasher>::from_sub_tree_store_configs_and_replicas(
                        tree_r_last_leafs,
                        &configs,
                        &replica_paths,
                    )?;

                Ok(TemporaryAuxCache {
                    labels: LabelsCache::new(&t_aux.labels).context("labels_cache")?,
                    tree_d,
                    tree_r_last: OctTreeData::OctLCTop(tree_r_last),
                    tree_r_last_config_levels,
                    tree_c: OctTreeData::OctTop(tree_c),
                    replica_path,
                    t_aux: t_aux.clone(),
                })
            }
            _ => panic!("Unsupported sector size"),
        }
         */
    }

    pub fn labels_for_layer(&self, layer: usize) -> &DiskStore<<Tree::Hasher as Hasher>::Domain> {
        self.labels.labels_for_layer(layer)
    }

    pub fn domain_node_at_layer(
        &self,
        layer: usize,
        node_index: u32,
    ) -> Result<<Tree::Hasher as Hasher>::Domain> {
        Ok(self.labels_for_layer(layer).read_at(node_index as usize)?)
    }

    pub fn column(&self, column_index: u32) -> Result<Column<Tree::Hasher>> {
        self.labels.column(column_index)
    }
}

type VerifyCallback = fn(&StoreConfig, usize) -> Result<()>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Labels<H: Hasher> {
    pub labels: Vec<StoreConfig>,
    pub _h: PhantomData<H>,
}

impl<H: Hasher> Labels<H> {
    pub fn new(labels: Vec<StoreConfig>) -> Self {
        Labels {
            labels,
            _h: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn verify_stores(&self, callback: VerifyCallback, cache_dir: &PathBuf) -> Result<()> {
        let updated_path_labels = self.labels.clone();
        for mut label in updated_path_labels {
            label.path = cache_dir.to_path_buf();
            callback(&label, BINARY_ARITY)?;
        }

        Ok(())
    }

    pub fn labels_for_layer(&self, layer: usize) -> Result<DiskStore<H::Domain>> {
        assert!(layer != 0, "Layer cannot be 0");
        assert!(
            layer <= self.layers(),
            "Layer {} is not available (only {} layers available)",
            layer,
            self.layers()
        );

        let row_index = layer - 1;
        let config = self.labels[row_index].clone();
        assert!(config.size.is_some());

        DiskStore::new_from_disk(config.size.unwrap(), OCT_ARITY, &config)
    }

    /// Returns label for the last layer.
    pub fn labels_for_last_layer(&self) -> Result<DiskStore<H::Domain>> {
        self.labels_for_layer(self.labels.len() - 1)
    }

    /// How many layers are available.
    fn layers(&self) -> usize {
        self.labels.len()
    }

    /// Build the column for the given node.
    pub fn column(&self, node: u32) -> Result<Column<H>> {
        let rows = self
            .labels
            .iter()
            .map(|label| {
                assert!(label.size.is_some());
                let store = DiskStore::new_from_disk(label.size.unwrap(), OCT_ARITY, &label)?;
                store.read_at(node as usize)
            })
            .collect::<Result<_>>()?;

        Column::new(node, rows)
    }

    /// Update all configs to the new passed in root cache path.
    pub fn update_root<P: AsRef<Path>>(&mut self, root: P) {
        for config in &mut self.labels {
            config.path = root.as_ref().into();
        }
    }
}

#[derive(Debug)]
pub struct LabelsCache<Tree: MerkleTreeTrait> {
    pub labels: Vec<DiskStore<<Tree::Hasher as Hasher>::Domain>>,
}

impl<Tree: MerkleTreeTrait> LabelsCache<Tree> {
    pub fn new(labels: &Labels<Tree::Hasher>) -> Result<Self> {
        let mut disk_store_labels: Vec<DiskStore<<Tree::Hasher as Hasher>::Domain>> =
            Vec::with_capacity(labels.len());
        for i in 0..labels.len() {
            disk_store_labels.push(labels.labels_for_layer(i + 1)?);
        }

        Ok(LabelsCache {
            labels: disk_store_labels,
        })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn labels_for_layer(&self, layer: usize) -> &DiskStore<<Tree::Hasher as Hasher>::Domain> {
        assert!(layer != 0, "Layer cannot be 0");
        assert!(
            layer <= self.layers(),
            "Layer {} is not available (only {} layers available)",
            layer,
            self.layers()
        );

        let row_index = layer - 1;
        &self.labels[row_index]
    }

    /// Returns the labels on the last layer.
    pub fn labels_for_last_layer(&self) -> Result<&DiskStore<<Tree::Hasher as Hasher>::Domain>> {
        Ok(&self.labels[self.labels.len() - 1])
    }

    /// How many layers are available.
    fn layers(&self) -> usize {
        self.labels.len()
    }

    /// Build the column for the given node.
    pub fn column(&self, node: u32) -> Result<Column<Tree::Hasher>> {
        let rows = self
            .labels
            .iter()
            .map(|labels| labels.read_at(node as usize))
            .collect::<Result<_>>()?;

        Column::new(node, rows)
    }
}

pub fn get_node<H: Hasher>(data: &[u8], index: usize) -> Result<H::Domain> {
    H::Domain::try_from_bytes(data_at_node(data, index).expect("invalid node math"))
}

/// Generate the replica id as expected for Stacked DRG.
pub fn generate_replica_id<H: Hasher, T: AsRef<[u8]>>(
    prover_id: &[u8; 32],
    sector_id: u64,
    ticket: &[u8; 32],
    comm_d: T,
) -> H::Domain {
    use sha2::{Digest, Sha256};

    let hash = Sha256::new()
        .chain(prover_id)
        .chain(&sector_id.to_be_bytes()[..])
        .chain(ticket)
        .chain(AsRef::<[u8]>::as_ref(&comm_d))
        .result();

    bytes_into_fr_repr_safe(hash.as_ref()).into()
}
