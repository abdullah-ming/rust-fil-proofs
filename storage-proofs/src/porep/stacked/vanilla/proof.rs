use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::PathBuf;

use log::{info, trace};
use merkletree::merkle::{get_merkle_tree_len, is_merkle_tree_size_valid};
use merkletree::store::{DiskStore, StoreConfig};
use rayon::prelude::*;

use super::{
    challenges::LayerChallenges,
    column::Column,
    create_label, create_label_exp,
    graph::StackedBucketGraph,
    hash::hash_single_column,
    params::{
        get_node, Labels, LabelsCache, PersistentAux, Proof, PublicInputs, PublicParams,
        ReplicaColumnProof, Tau, TemporaryAux, TemporaryAuxCache, TransformedLayers, BINARY_ARITY,
        OCT_ARITY,
    },
    EncodingProof, LabelingProof,
};

use crate::cache_key::CacheKey;
use crate::data::Data;
use crate::drgraph::Graph;
use crate::encode::{decode, encode};
use crate::error::Result;
use crate::hasher::{Domain, HashFunction, Hasher};
use crate::measurements::{
    measure_op,
    Operation::{CommD, EncodeWindowTimeAll},
};
use crate::merkle::*;
use crate::porep::PoRep;
use crate::util::NODE_SIZE;
pub const TOTAL_PARENTS: usize = 37;

#[derive(Debug)]
pub struct StackedDrg<'a, Tree: 'a + MerkleTreeTrait, G: 'a + Hasher> {
    _a: PhantomData<&'a Tree>,
    _b: PhantomData<&'a G>,
}

impl<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher> StackedDrg<'a, Tree, G> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prove_layers(
        graph: &StackedBucketGraph<Tree::Hasher>,
        pub_inputs: &PublicInputs<<Tree::Hasher as Hasher>::Domain, <G as Hasher>::Domain>,
        p_aux: &PersistentAux<<Tree::Hasher as Hasher>::Domain>,
        t_aux: &TemporaryAuxCache<Tree, G>,
        layer_challenges: &LayerChallenges,
        layers: usize,
        _total_layers: usize,
        partition_count: usize,
    ) -> Result<Vec<Vec<Proof<Tree, G>>>> {
        assert!(layers > 0);
        assert_eq!(t_aux.labels.len(), layers);

        let graph_size = graph.size();

        // Sanity checks on restored trees.
        assert!(pub_inputs.tau.is_some());
        assert_eq!(pub_inputs.tau.as_ref().unwrap().comm_d, t_aux.tree_d.root());

        let sector_size =
            (graph.size() * std::mem::size_of::<<Tree::Hasher as Hasher>::Domain>()) as u64;

        let get_drg_parents_columns = |x: usize| -> Result<Vec<Column<Tree::Hasher>>> {
            let base_degree = graph.base_graph().degree();

            let mut columns = Vec::with_capacity(base_degree);

            let mut parents = vec![0; base_degree];
            graph.base_parents(x, &mut parents)?;

            for parent in &parents {
                columns.push(t_aux.column(*parent)?);
            }

            debug_assert!(columns.len() == base_degree);

            Ok(columns)
        };

        let get_exp_parents_columns = |x: usize| -> Result<Vec<Column<Tree::Hasher>>> {
            let mut parents = vec![0; graph.expansion_degree()];
            graph.expanded_parents(x, &mut parents);

            parents.iter().map(|parent| t_aux.column(*parent)).collect()
        };

        (0..partition_count)
            .map(|k| {
                trace!("proving partition {}/{}", k + 1, partition_count);

                // Derive the set of challenges we are proving over.
                let challenges = pub_inputs.challenges(layer_challenges, graph_size, Some(k));
                let base_tree_leafs =
                    sector_size as usize / std::mem::size_of::<<Tree::Hasher as Hasher>::Domain>();

                // Stacked commitment specifics
                challenges
                    .into_par_iter()
                    .enumerate()
                    .map(|(challenge_index, challenge)| {
                        trace!(" challenge {} ({})", challenge, challenge_index);
                        assert!(challenge < graph.size(), "Invalid challenge");
                        assert!(challenge > 0, "Invalid challenge");

                        // Initial data layer openings (c_X in Comm_D)
                        let comm_d_proof = t_aux.tree_d.gen_proof(challenge)?;
                        assert!(comm_d_proof.validate(challenge));

                        // Stacked replica column openings
                        let rcp = {
                            let (c_x, drg_parents, exp_parents) = {
                                assert_eq!(p_aux.comm_c, t_aux.tree_c.root());
                                let tree_c = &t_aux.tree_c;

                                // All labels in C_X
                                trace!("  c_x");
                                let c_x = t_aux.column(challenge as u32)?.into_proof(tree_c)?;

                                // All labels in the DRG parents.
                                trace!("  drg_parents");
                                let drg_parents = get_drg_parents_columns(challenge)?
                                    .into_iter()
                                    .map(|column| column.into_proof(tree_c))
                                    .collect::<Result<_>>()?;

                                // Labels for the expander parents
                                trace!("  exp_parents");
                                let exp_parents = get_exp_parents_columns(challenge)?
                                    .into_iter()
                                    .map(|column| column.into_proof(tree_c))
                                    .collect::<Result<_>>()?;

                                (c_x, drg_parents, exp_parents)
                            };

                            ReplicaColumnProof {
                                c_x,
                                drg_parents,
                                exp_parents,
                            }
                        };

                        // Final replica layer openings
                        trace!("final replica layer openings");
                        let comm_r_last_proof = t_aux.tree_r_last.gen_proof(challenge)?;

                        // Labeling Proofs Layer 1..l
                        let mut labeling_proofs = HashMap::with_capacity(layers);
                        let mut encoding_proof = None;

                        for layer in 1..=layers {
                            trace!("  encoding proof layer {}", layer,);
                            let parents_data: Vec<<Tree::Hasher as Hasher>::Domain> = if layer == 1
                            {
                                let mut parents = vec![0; graph.base_graph().degree()];
                                graph.base_parents(challenge, &mut parents)?;

                                parents
                                    .into_iter()
                                    .map(|parent| t_aux.domain_node_at_layer(layer, parent))
                                    .collect::<Result<_>>()?
                            } else {
                                let mut parents = vec![0; graph.degree()];
                                graph.parents(challenge, &mut parents)?;
                                let base_parents_count = graph.base_graph().degree();

                                parents
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, parent)| {
                                        if i < base_parents_count {
                                            // parents data for base parents is from the current layer
                                            t_aux.domain_node_at_layer(layer, parent)
                                        } else {
                                            // parents data for exp parents is from the previous layer
                                            t_aux.domain_node_at_layer(layer - 1, parent)
                                        }
                                    })
                                    .collect::<Result<_>>()?
                            };

                            // repeat parents
                            let mut parents_data_full = vec![Default::default(); TOTAL_PARENTS];
                            for chunk in parents_data_full.chunks_mut(parents_data.len()) {
                                chunk.copy_from_slice(&parents_data[..chunk.len()]);
                            }

                            let proof = LabelingProof::<Tree::Hasher>::new(
                                challenge as u64,
                                parents_data_full.clone(),
                            );

                            {
                                let labeled_node = rcp.c_x.get_node_at_layer(layer)?;
                                assert!(
                                    proof.verify(&pub_inputs.replica_id, &labeled_node),
                                    format!("Invalid encoding proof generated at layer {}", layer)
                                );
                                trace!("Valid encoding proof generated at layer {}", layer);
                            }

                            labeling_proofs.insert(layer, proof);

                            if layer == layers {
                                encoding_proof =
                                    Some(EncodingProof::new(challenge as u64, parents_data_full));
                            }
                        }

                        Ok(Proof {
                            comm_d_proofs: comm_d_proof,
                            replica_column_proofs: rcp,
                            comm_r_last_proof,
                            labeling_proofs,
                            encoding_proof: encoding_proof.expect("invalid tapering"),
                        })
                    })
                    .collect()
            })
            .collect()
    }

    pub(crate) fn extract_and_invert_transform_layers(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        data: &mut [u8],
        config: StoreConfig,
    ) -> Result<()> {
        trace!("extract_and_invert_transform_layers");

        let layers = layer_challenges.layers();
        assert!(layers > 0);

        // generate labels
        let (labels, _) = Self::generate_labels(graph, layer_challenges, replica_id, config)?;

        let last_layer_labels = labels.labels_for_last_layer()?;
        let size = merkletree::store::Store::len(last_layer_labels);

        for (key, encoded_node_bytes) in last_layer_labels
            .read_range(0..size)?
            .into_iter()
            .zip(data.chunks_mut(NODE_SIZE))
        {
            let encoded_node =
                <Tree::Hasher as Hasher>::Domain::try_from_bytes(encoded_node_bytes)?;
            let data_node = decode::<<Tree::Hasher as Hasher>::Domain>(key, encoded_node);

            // store result in the data
            encoded_node_bytes.copy_from_slice(AsRef::<[u8]>::as_ref(&data_node));
        }

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn generate_labels(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        config: StoreConfig,
    ) -> Result<(LabelsCache<Tree>, Labels<Tree::Hasher>)> {
        info!("generate labels");

        let layers = layer_challenges.layers();
        // For now, we require it due to changes in encodings structure.
        let mut labels: Vec<DiskStore<<Tree::Hasher as Hasher>::Domain>> =
            Vec::with_capacity(layers);
        let mut label_configs: Vec<StoreConfig> = Vec::with_capacity(layers);

        let layer_size = graph.size() * NODE_SIZE;
        // NOTE: this means we currently keep 2x sector size around, to improve speed.
        let mut labels_buffer = vec![0u8; 2 * layer_size];

        for layer in 1..=layers {
            info!("generating layer: {}", layer);

            if layer == 1 {
                let layer_labels = &mut labels_buffer[..layer_size];
                for node in 0..graph.size() {
                    create_label(graph, replica_id, layer_labels, node)?;
                }
            } else {
                let (layer_labels, exp_labels) = labels_buffer.split_at_mut(layer_size);
                for node in 0..graph.size() {
                    create_label_exp(graph, replica_id, exp_labels, layer_labels, node)?;
                }
            }

            info!("  setting exp parents");
            labels_buffer.copy_within(..layer_size, layer_size);

            // Write the result to disk to avoid keeping it in memory all the time.
            let layer_config =
                StoreConfig::from_config(&config, CacheKey::label_layer(layer), Some(graph.size()));

            info!("  storing labels on disk");
            // Construct and persist the layer data.
            let layer_store: DiskStore<<Tree::Hasher as Hasher>::Domain> =
                DiskStore::new_from_slice_with_config(
                    graph.size(),
                    OCT_ARITY,
                    &labels_buffer[..layer_size],
                    layer_config.clone(),
                )?;
            info!(
                "  generated layer {} store with id {}",
                layer, layer_config.id
            );

            // Track the layer specific store and StoreConfig for later retrieval.
            labels.push(layer_store);
            label_configs.push(layer_config);
        }

        assert_eq!(
            labels.len(),
            layers,
            "Invalid amount of layers encoded expected"
        );

        Ok((
            LabelsCache::<Tree> { labels },
            Labels::<Tree::Hasher> {
                labels: label_configs,
                _h: PhantomData,
            },
        ))
    }

    fn build_binary_tree<K: Hasher>(
        tree_data: &[u8],
        config: StoreConfig,
    ) -> Result<BinaryMerkleTree<K>> {
        trace!("building tree (size: {})", tree_data.len());

        let leafs = tree_data.len() / NODE_SIZE;
        assert_eq!(tree_data.len() % NODE_SIZE, 0);

        let tree = MerkleTree::from_par_iter_with_config(
            (0..leafs)
                .into_par_iter()
                // TODO: proper error handling instead of `unwrap()`
                .map(|i| get_node::<K>(tree_data, i).unwrap()),
            config,
        )?;
        Ok(tree)
    }

    pub(crate) fn transform_and_replicate_layers(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        data: Data,
        data_tree: Option<BinaryMerkleTree<G>>,
        config: StoreConfig,
        replica_path: PathBuf,
    ) -> Result<TransformedLayers<Tree::Hasher, G>> {
        // Generate key layers.
        let (_, labels) = measure_op(EncodeWindowTimeAll, || {
            Self::generate_labels(graph, layer_challenges, replica_id, config.clone())
        })?;

        Self::transform_and_replicate_layers_inner(
            graph,
            layer_challenges,
            data,
            data_tree,
            config,
            replica_path,
            labels,
        )
    }

    /// Writes out replica data, possibly splitting out replicas based
    /// on how many sub-trees will require invidividual replica files.
    /// Returns a list of replica paths written.
    fn write_replica_data(
        data: &Data,
        tree_count: usize,
        leafs: usize,
        replica_path: &PathBuf,
    ) -> Result<Vec<PathBuf>> {
        use std::fs::OpenOptions;
        use std::io::prelude::*;
        use std::path::Path;

        if tree_count == 1 {
            // Store and persist encoded replica data.
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .open(&replica_path)?;

            // Only write the replica's base layer of leaf data.
            trace!("Writing replica data for {} nodes", leafs);

            f.write_all(&data.as_ref()[0..leafs * NODE_SIZE])?;

            Ok(vec![replica_path.to_path_buf()])
        } else {
            // Store and persist encoded replica data.
            let mut replica_paths = Vec::with_capacity(tree_count);
            let mut start = 0;
            let mut end = leafs * NODE_SIZE;

            for i in 0..tree_count {
                replica_paths.push(
                    Path::new(
                        format!("{:?}-{}", replica_path, i)
                            .replace("\"", "")
                            .as_str(),
                    )
                    .to_path_buf(),
                );
                let mut f = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .open(&replica_paths[i])?;

                // Only write the replica's base layer of leaf data.
                trace!(
                    "Writing replica data for {} nodes [{}-{}]",
                    leafs,
                    start,
                    end
                );
                f.write_all(&data.as_ref()[start..end])?;
                start = end;
                end += leafs * NODE_SIZE;
            }

            Ok(replica_paths)
        }
    }

    pub(crate) fn transform_and_replicate_layers_inner(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        mut data: Data,
        data_tree: Option<BinaryMerkleTree<G>>,
        config: StoreConfig,
        replica_path: PathBuf,
        label_configs: Labels<Tree::Hasher>,
    ) -> Result<TransformedLayers<Tree::Hasher, G>> {
        trace!("transform_and_replicate_layers");
        let nodes_count = graph.size();

        assert_eq!(data.len(), nodes_count * NODE_SIZE);
        trace!("nodes count {}, data len {}", nodes_count, data.len());

        let transform_and_replicate = |tree_count| -> Result<(_, _, _, _, _, _)> {
            let nodes_count = graph.size() / tree_count;

            // Ensure that the node count will work for binary and oct arities.
            trace!(
                "is_merkle_tree_size_valid({}, BINARY_ARITY) = {}",
                nodes_count,
                is_merkle_tree_size_valid(nodes_count, BINARY_ARITY)
            );
            trace!(
                "is_merkle_tree_size_valid({}, OCT_ARITY) = {}",
                nodes_count,
                is_merkle_tree_size_valid(nodes_count, OCT_ARITY)
            );
            assert!(is_merkle_tree_size_valid(nodes_count, BINARY_ARITY));
            assert!(is_merkle_tree_size_valid(nodes_count, OCT_ARITY));

            let layers = layer_challenges.layers();
            assert!(layers > 0);

            // Generate all store configs that we need based on the
            // cache_path in the specified config.
            let mut tree_d_config = StoreConfig::from_config(
                &config,
                CacheKey::CommDTree.to_string(),
                Some(get_merkle_tree_len(nodes_count, BINARY_ARITY)?),
            );
            tree_d_config.levels =
                StoreConfig::default_cached_above_base_layer(nodes_count, BINARY_ARITY);

            let mut tree_r_last_config = StoreConfig::from_config(
                &config,
                CacheKey::CommRLastTree.to_string(),
                Some(get_merkle_tree_len(nodes_count, OCT_ARITY)?),
            );
            tree_r_last_config.levels =
                StoreConfig::default_cached_above_base_layer(nodes_count, OCT_ARITY);

            let mut tree_c_config = StoreConfig::from_config(
                &config,
                CacheKey::CommCTree.to_string(),
                Some(get_merkle_tree_len(nodes_count, OCT_ARITY)?),
            );
            tree_c_config.levels =
                StoreConfig::default_cached_above_base_layer(nodes_count, OCT_ARITY);

            let labels = LabelsCache::<Tree>::new(&label_configs)?;

            // Build the tree for CommC
            // FIXME: Removed "measure_op(GenerateTreeC" closure, as it wasn't building
            let tree_c_root: Result<<Tree::Hasher as Hasher>::Domain> = {
                info!("Building column hashes");

                let configs = split_config(Some(tree_c_config.clone()), tree_count)?;
                let mut trees = Vec::with_capacity(tree_count);

                for i in 0..tree_count {
                    let mut hashes: Vec<<Tree::Hasher as Hasher>::Domain> =
                        vec![<Tree::Hasher as Hasher>::Domain::default(); nodes_count];

                    rayon::scope(|s| {
                        // spawn n = num_cpus * 2 threads
                        let n = num_cpus::get() * 2;

                        // only split if we have at least two elements per thread
                        let num_chunks = if n > nodes_count * 2 { 1 } else { n };

                        // chunk into n chunks
                        let chunk_size = (nodes_count as f64 / num_chunks as f64).ceil() as usize;

                        // calculate all n chunks in parallel
                        for (chunk, hashes_chunk) in hashes.chunks_mut(chunk_size).enumerate() {
                            let labels = &labels;

                            s.spawn(move |_| {
                                for (j, hash) in hashes_chunk.iter_mut().enumerate() {
                                    let data: Vec<_> = (1..=layers)
                                        .map(|layer| {
                                            let store = labels.labels_for_layer(layer);
                                            let el: <Tree::Hasher as Hasher>::Domain = store
                                                .read_at((i * nodes_count) + j + chunk * chunk_size)
                                                .unwrap();
                                            el.into()
                                        })
                                        .collect();

                                    *hash = hash_single_column(&data).into();
                                }
                            });
                        }
                    });

                    info!("building tree_c");
                    assert!(configs[i].is_some());
                    trees.push(OctMerkleTree::<Tree::Hasher>::from_par_iter_with_config(
                        hashes.into_par_iter(),
                        if tree_count == 1 {
                            tree_c_config.clone()
                        } else {
                            configs[i].as_ref().unwrap().clone()
                        },
                    )?);
                }

                assert_eq!(tree_count, trees.len());

                if tree_count == 1 {
                    let tree_c = &trees[0];
                    tree_c_config.size = Some(tree_c.len());

                    Ok(tree_c.root())
                } else if tree_count == 16 {
                    tree_c_config.size = Some(trees[0].len());

                    // Build a top level tree consisting of sub_tree_count (i.e. 2) sub_trees, each of (tree_count / sub_tree_count) base layer trees.
                    let tree_c = OctTopMerkleTree::<Tree::Hasher>::from_sub_trees_as_trees(trees)?;

                    Ok(tree_c.root())
                } else {
                    assert!(tree_count == 2 || tree_count == 8 || tree_count == 16);
                    tree_c_config.size = Some(trees[0].len());
                    let tree_c = OctSubMerkleTree::<Tree::Hasher>::from_trees(trees)?;

                    Ok(tree_c.root())
                }
            };
            info!("tree_c done");

            // Build the MerkleTree over the original data (if needed).
            let tree_d = match data_tree {
                Some(t) => {
                    trace!("using existing original data merkle tree");
                    assert_eq!(t.len(), 2 * (data.len() / NODE_SIZE) - 1);

                    t
                }
                None => {
                    trace!("building merkle tree for the original data");
                    data.ensure_data()?;
                    measure_op(CommD, || {
                        Self::build_binary_tree::<G>(data.as_ref(), tree_d_config.clone())
                    })?
                }
            };
            tree_d_config.size = Some(tree_d.len());
            assert_eq!(tree_d_config.size.unwrap(), tree_d.len());
            let tree_d_root = tree_d.root();
            drop(tree_d);

            // Encode original data into the last layer.
            info!("building tree_r_last");
            // FIXME: Removed "measure_op(GenerateTreeRLast" closure, as it wasn't building
            let tree_r_last_root: Result<<Tree::Hasher as Hasher>::Domain> = {
                data.ensure_data()?;

                let last_layer_labels = labels.labels_for_last_layer()?;
                let size = Store::len(last_layer_labels);

                let mut trees = Vec::with_capacity(tree_count);
                let mut start = 0;
                let mut end = size / tree_count;

                let configs = split_config(Some(tree_r_last_config.clone()), tree_count)?;
                for config in &configs {
                    let encoded_data = last_layer_labels
                        .read_range(start..end)?
                        .into_par_iter()
                        .zip(
                            data.as_mut()[(start * NODE_SIZE)..(end * NODE_SIZE)]
                                .par_chunks_mut(NODE_SIZE),
                        )
                        .map(|(key, data_node_bytes)| {
                            let data_node =
                                <Tree::Hasher as Hasher>::Domain::try_from_bytes(data_node_bytes)
                                    .unwrap();
                            let encoded_node =
                                encode::<<Tree::Hasher as Hasher>::Domain>(key, data_node);
                            data_node_bytes.copy_from_slice(AsRef::<[u8]>::as_ref(&encoded_node));

                            encoded_node
                        });

                    assert!(config.is_some());
                    trees.push(OctLCMerkleTree::<Tree::Hasher>::from_par_iter_with_config(
                        encoded_data,
                        if tree_count == 1 {
                            tree_r_last_config.clone()
                        } else {
                            config.as_ref().unwrap().clone()
                        },
                    )?);

                    start = end;
                    end += size / tree_count;
                }

                if tree_count == 1 {
                    // In this case, tree_r_last is already complete.
                    let tree_r_last = &trees[0];
                    tree_r_last_config.size = Some(tree_r_last.len());
                    assert_eq!(tree_r_last_config.size.unwrap(), tree_r_last.len());

                    Self::write_replica_data(
                        &data,
                        tree_count,
                        tree_r_last.leafs(),
                        &replica_path,
                    )?;

                    Ok(tree_r_last.root())
                } else if tree_count == 16 {
                    // Note: Given the shape of OctLCTopTree, a tree count
                    // of 16 will yield 2 top layer trees, each consisting
                    // of 8 sub layer trees.
                    let leafs = trees[0].leafs();
                    assert_eq!((leafs * NODE_SIZE) * tree_count, data.len());
                    tree_r_last_config.size = Some(trees[0].len());

                    let unwrapped_configs = {
                        let mut c = Vec::with_capacity(tree_count);
                        for i in 0..tree_count {
                            c.push(configs[i].as_ref().unwrap().clone());
                            c[i].levels = configs[i].as_ref().unwrap().levels;
                        }

                        c
                    };
                    let replica_paths =
                        Self::write_replica_data(&data, tree_count, leafs, &replica_path)?;

                    let tree_r_last =
                        OctLCTopMerkleTree::<Tree::Hasher>::from_sub_tree_store_configs_and_replicas(
                            leafs,
                            &unwrapped_configs,
                            &replica_paths,
                        )?;

                    Ok(tree_r_last.root())
                } else {
                    // In this case, construct the compound tree_r_last from the sub-trees we have.
                    assert!(tree_count == 2 || tree_count == 8);
                    let leafs = trees[0].leafs();
                    assert_eq!((leafs * NODE_SIZE) * tree_count, data.len());
                    tree_r_last_config.size = Some(trees[0].len());

                    let unwrapped_configs = {
                        let mut c = Vec::with_capacity(tree_count);
                        for i in 0..tree_count {
                            c.push(configs[i].as_ref().unwrap().clone());
                            c[i].levels = configs[i].as_ref().unwrap().levels;
                        }

                        c
                    };
                    let replica_paths =
                        Self::write_replica_data(&data, tree_count, leafs, &replica_path)?;

                    let tree_r_last =
                        OctLCSubMerkleTree::<Tree::Hasher>::from_store_configs_and_replicas(
                            leafs,
                            &unwrapped_configs,
                            &replica_paths,
                        )?;

                    Ok(tree_r_last.root())
                }
            };
            info!("tree_r_last done");

            data.drop_data();

            Ok((
                tree_d_config,
                tree_c_config,
                tree_r_last_config,
                tree_d_root,
                tree_c_root?,
                tree_r_last_root?,
            ))
        };

        let sector_size = (nodes_count * NODE_SIZE) as u64;
        let (
            tree_d_config,
            tree_c_config,
            tree_r_last_config,
            tree_d_root,
            tree_c_root,
            tree_r_last_root,
        ) = match sector_size {
            SECTOR_SIZE_2_KIB | SECTOR_SIZE_8_MIB | SECTOR_SIZE_512_MIB => {
                transform_and_replicate(1)
            }
            SECTOR_SIZE_4_KIB | SECTOR_SIZE_16_MIB | SECTOR_SIZE_1_GIB => {
                transform_and_replicate(2)
            }
            SECTOR_SIZE_32_GIB => transform_and_replicate(8),
            SECTOR_SIZE_32_KIB | SECTOR_SIZE_64_GIB => transform_and_replicate(16),
            _ => panic!("Unsupported data len"),
        }?;

        // comm_r = H(comm_c || comm_r_last)
        let comm_r: <Tree::Hasher as Hasher>::Domain =
            <Tree::Hasher as Hasher>::Function::hash2(&tree_c_root, &tree_r_last_root);

        Ok((
            Tau {
                comm_d: tree_d_root,
                comm_r,
            },
            PersistentAux {
                comm_c: tree_c_root,
                comm_r_last: tree_r_last_root,
            },
            TemporaryAux {
                labels: label_configs,
                tree_d_config,
                tree_r_last_config,
                tree_c_config,
                _g: PhantomData,
            },
        ))
    }

    /// Phase1 of replication.
    pub fn replicate_phase1(
        pp: &'a PublicParams<Tree::Hasher>,
        replica_id: &<Tree::Hasher as Hasher>::Domain,
        config: StoreConfig,
    ) -> Result<Labels<Tree::Hasher>> {
        info!("replicate_phase1");

        let (_, labels) = measure_op(EncodeWindowTimeAll, || {
            Self::generate_labels(&pp.graph, &pp.layer_challenges, replica_id, config)
        })?;

        Ok(labels)
    }

    #[allow(clippy::type_complexity)]
    /// Phase2 of replication.
    #[allow(clippy::type_complexity)]
    pub fn replicate_phase2(
        pp: &'a PublicParams<Tree::Hasher>,
        labels: Labels<Tree::Hasher>,
        data: Data<'a>,
        data_tree: BinaryMerkleTree<G>,
        config: StoreConfig,
        replica_path: PathBuf,
    ) -> Result<(
        <Self as PoRep<'a, Tree::Hasher, G>>::Tau,
        <Self as PoRep<'a, Tree::Hasher, G>>::ProverAux,
    )> {
        info!("replicate_phase2");

        let (tau, paux, taux) = Self::transform_and_replicate_layers_inner(
            &pp.graph,
            &pp.layer_challenges,
            data,
            Some(data_tree),
            config,
            replica_path,
            labels,
        )?;

        Ok((tau, (paux, taux)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ff::Field;
    use paired::bls12_381::Fr;
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    use crate::drgraph::{new_seed, BASE_DEGREE};
    use crate::fr32::fr_into_bytes;
    use crate::hasher::{Blake2sHasher, PedersenHasher, PoseidonHasher, Sha256Hasher};
    use crate::merkle::{BinaryMerkleTree, MerkleTreeTrait};
    use crate::porep::stacked::{PrivateInputs, SetupParams, EXP_DEGREE};
    use crate::porep::PoRep;
    use crate::proof::ProofScheme;

    const DEFAULT_STACKED_LAYERS: usize = 11;

    #[test]
    fn test_calculate_fixed_challenges() {
        let layer_challenges = LayerChallenges::new(10, 333);
        let expected = 333;

        let calculated_count = layer_challenges.challenges_count_all();
        assert_eq!(expected as usize, calculated_count);
    }

    #[test]
    fn extract_all_pedersen() {
        test_extract_all::<BinaryMerkleTree<PedersenHasher>>();
    }

    #[test]
    fn extract_all_sha256() {
        test_extract_all::<BinaryMerkleTree<Sha256Hasher>>();
    }

    #[test]
    fn extract_all_blake2s() {
        test_extract_all::<BinaryMerkleTree<Blake2sHasher>>();
    }

    #[test]
    fn extract_all_poseidon() {
        test_extract_all::<BinaryMerkleTree<PoseidonHasher>>();
    }

    fn test_extract_all<Tree: 'static + MerkleTreeTrait>() {
        // femme::pretty::Logger::new()
        //     .start(log::LevelFilter::Trace)
        //     .ok();

        let rng = &mut XorShiftRng::from_seed(crate::TEST_SEED);
        let replica_id: <Tree::Hasher as Hasher>::Domain =
            <Tree::Hasher as Hasher>::Domain::random(rng);
        let nodes = 64;

        let data: Vec<u8> = (0..nodes)
            .flat_map(|_| {
                let v: <Tree::Hasher as Hasher>::Domain =
                    <Tree::Hasher as Hasher>::Domain::random(rng);
                v.into_bytes()
            })
            .collect();

        let challenges = LayerChallenges::new(DEFAULT_STACKED_LAYERS, 5);

        // create a copy, so we can compare roundtrips
        let mut data_copy = data.clone();

        let sp = SetupParams {
            nodes,
            degree: BASE_DEGREE,
            expansion_degree: EXP_DEGREE,
            seed: new_seed(),
            layer_challenges: challenges.clone(),
        };

        let pp = StackedDrg::<Tree, Blake2sHasher>::setup(&sp).expect("setup failed");

        // MT for original data is always named tree-d, and it will be
        // referenced later in the process as such.
        let cache_dir = tempfile::tempdir().unwrap();
        let config = StoreConfig::new(
            cache_dir.path(),
            CacheKey::CommDTree.to_string(),
            StoreConfig::default_cached_above_base_layer(nodes, BINARY_ARITY),
        );

        // Generate a replica path.
        let temp_dir = tempdir::TempDir::new("test-extract-all").unwrap();
        let temp_path = temp_dir.path();
        let replica_path = temp_path.join("replica-path");

        StackedDrg::<Tree, Blake2sHasher>::replicate(
            &pp,
            &replica_id,
            (&mut data_copy[..]).into(),
            None,
            config.clone(),
            replica_path.clone(),
        )
        .expect("replication failed");

        assert_ne!(data, data_copy);

        let decoded_data = StackedDrg::<Tree, Blake2sHasher>::extract_all(
            &pp,
            &replica_id,
            data_copy.as_mut_slice(),
            Some(config.clone()),
        )
        .expect("failed to extract data");

        assert_eq!(data, decoded_data);
    }

    fn prove_verify_fixed(n: usize) {
        let challenges = LayerChallenges::new(DEFAULT_STACKED_LAYERS, 5);

        test_prove_verify::<BinaryMerkleTree<PedersenHasher>>(n, challenges.clone());
        test_prove_verify::<BinaryMerkleTree<Sha256Hasher>>(n, challenges.clone());
        test_prove_verify::<BinaryMerkleTree<Blake2sHasher>>(n, challenges.clone());
        test_prove_verify::<BinaryMerkleTree<PoseidonHasher>>(n, challenges.clone());
    }

    fn test_prove_verify<Tree: 'static + MerkleTreeTrait>(n: usize, challenges: LayerChallenges) {
        // This will be called multiple times, only the first one succeeds, and that is ok.
        // femme::pretty::Logger::new()
        //     .start(log::LevelFilter::Trace)
        //     .ok();

        let rng = &mut XorShiftRng::from_seed(crate::TEST_SEED);

        let degree = BASE_DEGREE;
        let expansion_degree = EXP_DEGREE;
        let replica_id: <Tree::Hasher as Hasher>::Domain =
            <Tree::Hasher as Hasher>::Domain::random(rng);
        let data: Vec<u8> = (0..n)
            .flat_map(|_| fr_into_bytes(&Fr::random(rng)))
            .collect();

        // create a copy, so we can compare roundtrips
        let mut data_copy = data.clone();
        let partitions = 2;

        let sp = SetupParams {
            nodes: n,
            degree,
            expansion_degree,
            seed: new_seed(),
            layer_challenges: challenges.clone(),
        };

        // MT for original data is always named tree-d, and it will be
        // referenced later in the process as such.
        let cache_dir = tempfile::tempdir().unwrap();
        let config = StoreConfig::new(
            cache_dir.path(),
            CacheKey::CommDTree.to_string(),
            StoreConfig::default_cached_above_base_layer(n, BINARY_ARITY),
        );

        // Generate a replica_path.
        let temp_dir = tempdir::TempDir::new("test-prove-verify").unwrap();
        let temp_path = temp_dir.path();
        let replica_path = temp_path.join("replica-path");

        let pp = StackedDrg::<Tree, Blake2sHasher>::setup(&sp).expect("setup failed");
        let (tau, (p_aux, t_aux)) = StackedDrg::<Tree, Blake2sHasher>::replicate(
            &pp,
            &replica_id,
            (&mut data_copy[..]).into(),
            None,
            config,
            replica_path.clone(),
        )
        .expect("replication failed");
        assert_ne!(data, data_copy);

        let seed = rng.gen();

        let pub_inputs =
            PublicInputs::<<Tree::Hasher as Hasher>::Domain, <Blake2sHasher as Hasher>::Domain> {
                replica_id,
                seed,
                tau: Some(tau),
                k: None,
            };

        // Store a copy of the t_aux for later resource deletion.
        let t_aux_orig = t_aux.clone();

        // Convert TemporaryAux to TemporaryAuxCache, which instantiates all
        // elements based on the configs stored in TemporaryAux.
        let t_aux = TemporaryAuxCache::<Tree, Blake2sHasher>::new(&t_aux, replica_path)
            .expect("failed to restore contents of t_aux");

        let priv_inputs = PrivateInputs { p_aux, t_aux };

        let all_partition_proofs = &StackedDrg::<Tree, Blake2sHasher>::prove_all_partitions(
            &pp,
            &pub_inputs,
            &priv_inputs,
            partitions,
        )
        .expect("failed to generate partition proofs");

        let proofs_are_valid = StackedDrg::<Tree, Blake2sHasher>::verify_all_partitions(
            &pp,
            &pub_inputs,
            all_partition_proofs,
        )
        .expect("failed to verify partition proofs");

        // Discard cached MTs that are no longer needed.
        TemporaryAux::<Tree::Hasher, Blake2sHasher>::clear_temp(t_aux_orig)
            .expect("t_aux delete failed");

        assert!(proofs_are_valid);
    }

    table_tests! {
        prove_verify_fixed {
           prove_verify_fixed_64_64(64);
        }
    }

    #[test]
    // We are seeing a bug, in which setup never terminates for some sector sizes.
    // This test is to debug that and should remain as a regression teset.
    fn setup_terminates() {
        let degree = BASE_DEGREE;
        let expansion_degree = EXP_DEGREE;
        let nodes = 1024 * 1024 * 32 * 8; // This corresponds to 8GiB sectors (32-byte nodes)
        let layer_challenges = LayerChallenges::new(10, 333);
        let sp = SetupParams {
            nodes,
            degree,
            expansion_degree,
            seed: new_seed(),
            layer_challenges: layer_challenges.clone(),
        };

        // When this fails, the call to setup should panic, but seems to actually hang (i.e. neither return nor panic) for some reason.
        // When working as designed, the call to setup returns without error.
        let _pp = StackedDrg::<BinaryMerkleTree<PedersenHasher>, Blake2sHasher>::setup(&sp)
            .expect("setup failed");
    }
}
