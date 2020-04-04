use std::marker::PhantomData;

use bellperson::gadgets::num;
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use generic_array::typenum::{self, marker_traits::Unsigned, U8};
use paired::bls12_381::{Bls12, Fr};

use crate::compound_proof::CircuitComponent;
use crate::drgraph::graph_height;
use crate::error::Result;
use crate::gadgets::constraint;
use crate::gadgets::por::PoRCircuit;
use crate::gadgets::variables::Root;
use crate::hasher::{HashFunction, Hasher};
use crate::merkle::{DiskStore, MerkleProofTrait, MerkleTreeWrapper};
use crate::util::NODE_SIZE;

use super::vanilla::{PublicParams, PublicSector, SectorProof};

/// This is the `FallbackPoSt` circuit.
pub struct FallbackPoStCircuit<H: Hasher> {
    pub prover_id: Option<Fr>,
    pub sectors: Vec<Sector<H>>,
}

#[derive(Clone)]
pub struct Sector<H: Hasher> {
    pub comm_r: Option<Fr>,
    pub comm_c: Option<Fr>,
    pub comm_r_last: Option<Fr>,
    pub leafs: Vec<Option<Fr>>,
    #[allow(clippy::type_complexity)]
    pub paths: Vec<Vec<(Vec<Option<Fr>>, Option<usize>)>>,
    pub id: Option<Fr>,
    pub _h: PhantomData<H>,
}

impl<H: Hasher> Sector<H> {
    pub fn circuit<P: MerkleProofTrait<Hasher = H>>(
        sector: &PublicSector<H::Domain>,
        vanilla_proof: &SectorProof<P>,
    ) -> Result<Self> {
        let leafs = vanilla_proof
            .leafs()
            .iter()
            .map(|l| Some((*l).into()))
            .collect();

        let paths = vanilla_proof
            .paths()
            .iter()
            .map(|p| {
                p.iter()
                    .map(|v| {
                        (
                            v.0.iter().copied().map(Into::into).map(Some).collect(),
                            Some(v.1),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(Sector {
            leafs,
            id: Some(sector.id.into()),
            comm_r: Some(sector.comm_r.into()),
            comm_c: Some(vanilla_proof.comm_c.into()),
            comm_r_last: Some(vanilla_proof.comm_r_last.into()),
            paths,
            _h: PhantomData,
        })
    }

    pub fn blank_circuit(pub_params: &PublicParams) -> Self {
        let challenges_count = pub_params.challenge_count;
        let height = graph_height::<U8>(pub_params.sector_size as usize / NODE_SIZE);

        let leafs = vec![None; challenges_count];
        let paths =
            vec![vec![(vec![None; U8::to_usize() - 1], None); height - 1]; challenges_count];

        Sector {
            id: None,
            comm_r: None,
            comm_c: None,
            comm_r_last: None,
            leafs,
            paths,
            _h: PhantomData,
        }
    }
}

impl<H: 'static + Hasher> Circuit<Bls12> for &Sector<H> {
    fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let Sector {
            comm_r,
            comm_c,
            comm_r_last,
            leafs,
            paths,
            ..
        } = self;

        assert_eq!(paths.len(), leafs.len());

        // 1. Verify comm_r
        let comm_r_last_num = num::AllocatedNum::alloc(cs.namespace(|| "comm_r_last"), || {
            comm_r_last
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        let comm_c_num = num::AllocatedNum::alloc(cs.namespace(|| "comm_c"), || {
            comm_c
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        let comm_r_num = num::AllocatedNum::alloc(cs.namespace(|| "comm_r"), || {
            comm_r
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        comm_r_num.inputize(cs.namespace(|| "comm_r_input"))?;

        // 1. Verify H(Comm_C || comm_r_last) == comm_r
        {
            let hash_num = H::Function::hash2_circuit(
                cs.namespace(|| "H_comm_c_comm_r_last"),
                &comm_c_num,
                &comm_r_last_num,
            )?;

            // Check actual equality
            constraint::equal(
                cs,
                || "enforce_comm_c_comm_r_last_hash_comm_r",
                &comm_r_num,
                &hash_num,
            );
        }

        // 2. Verify Inclusion Paths
        for (i, (leaf, path)) in leafs.iter().zip(paths.iter()).enumerate() {
            PoRCircuit::<
                MerkleTreeWrapper<H, DiskStore<H::Domain>, typenum::U8, typenum::U0, typenum::U0>,
            >::synthesize(
                cs.namespace(|| format!("challenge_inclusion_{}", i)),
                Root::Val(*leaf),
                path.clone(),
                Root::from_allocated::<CS>(comm_r_last_num.clone()),
                true,
            )?;
        }

        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct ComponentPrivateInputs {}

impl<H: Hasher> CircuitComponent for FallbackPoStCircuit<H> {
    type ComponentPrivateInputs = ComponentPrivateInputs;
}

impl<H: 'static + Hasher> Circuit<Bls12> for FallbackPoStCircuit<H> {
    fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        for (i, sector) in self.sectors.iter().enumerate() {
            let cs = &mut cs.namespace(|| format!("sector_{}", i));

            sector.synthesize(cs)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ff::Field;
    use merkletree::store::StoreConfig;
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    use crate::compound_proof::CompoundProof;
    use crate::drgraph::{new_seed, BucketGraph, Graph, BASE_DEGREE};
    use crate::fr32::fr_into_bytes;
    use crate::gadgets::TestConstraintSystem;
    use crate::hasher::{Domain, HashFunction, Hasher, PedersenHasher, PoseidonHasher};
    use crate::merkle::{generate_tree, OctMerkleTree};
    use crate::porep::stacked::OCT_ARITY;
    use crate::post::fallback::{
        self, FallbackPoSt, FallbackPoStCompound, PrivateInputs, PrivateSector, PublicInputs,
        PublicSector,
    };
    use crate::proof::ProofScheme;
    use crate::util::NODE_SIZE;

    #[test]
    fn fallback_post_pedersen_single_partition_matching() {
        fallback_post::<PedersenHasher>(3, 3, 1, 19, 294_459);
    }

    #[test]
    fn fallback_post_poseidon_single_partition_matching() {
        fallback_post::<PoseidonHasher>(3, 3, 1, 19, 17_988);
    }

    #[test]
    fn fallback_post_poseidon_single_partition_smaller() {
        fallback_post::<PoseidonHasher>(2, 3, 1, 19, 17_988);
    }

    #[test]
    fn fallback_post_poseidon_two_partitions_matching() {
        fallback_post::<PoseidonHasher>(4, 2, 2, 13, 11_992);
    }

    #[test]
    fn fallback_post_poseidon_two_partitions_smaller() {
        fallback_post::<PoseidonHasher>(5, 3, 2, 19, 17_988);
    }

    #[test]
    #[ignore]
    fn metric_fallback_post_circuit_poseidon() {
        use crate::gadgets::BenchCS;

        let params = fallback::SetupParams {
            sector_size: 1024 * 1024 * 1024 * 32 as u64,
            challenge_count: 10,
            sector_count: 5,
        };

        let pp = FallbackPoSt::<OctMerkleTree<PoseidonHasher>>::setup(&params).unwrap();

        let mut cs = BenchCS::<Bls12>::new();
        FallbackPoStCompound::<OctMerkleTree<PoseidonHasher>>::blank_circuit(&pp)
            .synthesize(&mut cs)
            .unwrap();

        assert_eq!(cs.num_constraints(), 285_180);
    }

    fn fallback_post<H: 'static + Hasher>(
        total_sector_count: usize,
        sector_count: usize,
        partitions: usize,
        expected_num_inputs: usize,
        expected_constraints: usize,
    ) {
        use std::fs::File;
        use std::io::prelude::*;

        let rng = &mut XorShiftRng::from_seed(crate::TEST_SEED);

        let leaves = 64;
        let sector_size = leaves * NODE_SIZE;
        let randomness = H::Domain::random(rng);
        let prover_id = H::Domain::random(rng);

        let pub_params = fallback::PublicParams {
            sector_size: sector_size as u64,
            challenge_count: 5,
            sector_count,
        };

        // Construct and store an MT using a named DiskStore.
        let temp_dir = tempdir::TempDir::new("level_cache_tree_v1").unwrap();
        let temp_path = temp_dir.path();
        let config = StoreConfig::new(
            &temp_path,
            String::from("test-lc-tree"),
            StoreConfig::default_cached_above_base_layer(leaves as usize, OCT_ARITY),
        );

        let mut pub_sectors = Vec::new();
        let mut priv_sectors = Vec::new();
        let mut trees = Vec::new();

        for i in 0..total_sector_count {
            let (_data, tree) =
                generate_tree::<OctMerkleTree<H>, _>(rng, leaves, Some(temp_path.to_path_buf()));
            trees.push(tree);
        }

        for (i, tree) in trees.iter().enumerate() {
            let comm_c = H::Domain::random(rng);
            let comm_r_last = tree.root();

            priv_sectors.push(PrivateSector {
                tree,
                comm_c,
                comm_r_last,
            });

            let comm_r = H::Function::hash2(&comm_c, &comm_r_last);
            pub_sectors.push(PublicSector {
                id: (i as u64).into(),
                comm_r,
            });
        }

        let pub_inputs = PublicInputs {
            randomness,
            prover_id,
            sectors: &pub_sectors,
            k: None,
        };

        let priv_inputs = PrivateInputs::<OctMerkleTree<H>> {
            sectors: &priv_sectors,
        };

        let proofs = FallbackPoSt::<OctMerkleTree<H>>::prove_all_partitions(
            &pub_params,
            &pub_inputs,
            &priv_inputs,
            partitions,
        )
        .expect("proving failed");
        assert_eq!(proofs.len(), partitions);

        let is_valid = FallbackPoSt::<OctMerkleTree<H>>::verify_all_partitions(
            &pub_params,
            &pub_inputs,
            &proofs,
        )
        .expect("verification failed");
        assert!(is_valid);

        // actual circuit test

        //     let paths = proof
        //     .paths()
        //     .iter()
        //     .map(|p| {
        //         p.iter()
        //             .map(|v| {
        //                 (
        //                     v.0.iter().copied().map(Into::into).map(Some).collect(),
        //                     Some(v.1),
        //                 )
        //             })
        //             .collect::<Vec<_>>()
        //     })
        //     .collect();
        // let leafs = proof.leafs().iter().map(|l| Some((*l).into())).collect();
        //     (
        //         Some(pub_sectors[i].id.into()),
        //         Some(pub_sectors[i].comm_r.into()),
        //         Some(priv_sectors[i].comm_c.into()),
        //         Some(priv_sectors[i].comm_r_last.into()),
        //     )
        // Sector {
        //     params: &*JJ_PARAMS,
        //     id,
        //     leafs,
        //     paths,
        //     comm_r,
        //     comm_c,
        //     comm_r_last,
        //     _h: PhantomData,
        // }

        for (j, proof) in proofs.iter().enumerate() {
            // iterates over each partition
            let circuit_sectors = proof
                .sectors
                .iter()
                .enumerate()
                .map(|(i, proof)| {
                    // index into sectors by the correct offset
                    let i = j * sector_count + i;

                    if i < pub_sectors.len() {
                        Sector::circuit(&pub_sectors[i], proof)
                    } else {
                        // duplicated last one
                        let k = pub_sectors.len() - 1;
                        Sector::circuit(&pub_sectors[k], proof)
                    }
                })
                .collect::<Result<_>>()
                .unwrap();

            let mut cs = TestConstraintSystem::<Bls12>::new();

            let instance = FallbackPoStCircuit::<H> {
                sectors: circuit_sectors,
                prover_id: Some(prover_id.into()),
            };

            instance
                .synthesize(&mut cs)
                .expect("failed to synthesize circuit");

            assert!(cs.is_satisfied(), "constraints not satisfied");

            assert_eq!(
                cs.num_inputs(),
                expected_num_inputs,
                "wrong number of inputs"
            );
            assert_eq!(
                cs.num_constraints(),
                expected_constraints,
                "wrong number of constraints"
            );
            assert_eq!(cs.get_input(0, "ONE"), Fr::one());

            let generated_inputs =
                FallbackPoStCompound::<OctMerkleTree<H>>::generate_public_inputs(
                    &pub_inputs,
                    &pub_params,
                    Some(j),
                )
                .unwrap();
            let expected_inputs = cs.get_inputs();

            for ((input, label), generated_input) in
                expected_inputs.iter().skip(1).zip(generated_inputs.iter())
            {
                assert_eq!(input, generated_input, "{}", label);
            }

            assert_eq!(
                generated_inputs.len(),
                expected_inputs.len() - 1,
                "inputs are not the same length"
            );

            assert!(
                cs.verify(&generated_inputs),
                "verification failed with TestContraintSystem and generated inputs"
            );
        }
    }
}
