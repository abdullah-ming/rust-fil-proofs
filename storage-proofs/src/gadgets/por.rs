use std::marker::PhantomData;

use anyhow::ensure;
use bellperson::gadgets::boolean::{AllocatedBit, Boolean};
use bellperson::gadgets::{multipack, num};
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use generic_array::typenum::{self, Unsigned};
use paired::bls12_381::{Bls12, Fr};

use crate::compound_proof::{CircuitComponent, CompoundProof};
use crate::drgraph::graph_height;
use crate::error::Result;
use crate::gadgets::constraint;
use crate::gadgets::insertion::insert;
use crate::gadgets::variables::Root;
use crate::hasher::{HashFunction, Hasher, PoseidonArity};
use crate::merkle::{compound_path_length, MerkleProofTrait, MerkleTreeTrait};
use crate::parameter_cache::{CacheableParameters, ParameterSetMetadata};
use crate::por::PoR;
use crate::proof::ProofScheme;

/// Proof of retrievability.
///
/// # Fields
///
/// * `params` - The params for the bls curve.
/// * `value` - The value of the leaf.
/// * `auth_path` - The authentication path of the leaf in the tree.
/// * `root` - The merkle root of the tree.
///
pub struct PoRCircuit<Tree: MerkleTreeTrait> {
    value: Root<Bls12>,
    #[allow(clippy::type_complexity)]
    auth_path: AuthPath<Tree>,
    root: Root<Bls12>,
    private: bool,
    _tree: PhantomData<Tree>,
}

#[derive(Debug, Clone)]
struct AuthPath<Tree: MerkleTreeTrait> {
    base: SubPath<Tree::Hasher, Tree::Arity>,
    sub: SubPath<Tree::Hasher, Tree::SubTreeArity>,
    top: SubPath<Tree::Hasher, Tree::TopTreeArity>,
    _t: PhantomData<Tree>,
}

impl<Tree: MerkleTreeTrait> From<Vec<(Vec<Option<Fr>>, Option<usize>)>> for AuthPath<Tree> {
    fn from(mut base_opts: Vec<(Vec<Option<Fr>>, Option<usize>)>) -> Self {
        let has_top = Tree::TopTreeArity::to_usize() > 0;
        let has_sub = Tree::SubTreeArity::to_usize() > 0;
        let len = base_opts.len();

        let x = if has_top {
            2
        } else if has_sub {
            1
        } else {
            0
        };
        let mut opts = base_opts.split_off(len - x);

        let base = base_opts
            .into_iter()
            .map(|(hashes, index)| PathElement {
                hashes,
                index,
                _a: Default::default(),
                _h: Default::default(),
            })
            .collect();

        let top = if has_top {
            let (hashes, index) = opts.pop().unwrap();
            vec![PathElement {
                hashes,
                index,
                _a: Default::default(),
                _h: Default::default(),
            }]
        } else {
            Vec::new()
        };

        let sub = if has_sub {
            let (hashes, index) = opts.pop().unwrap();
            vec![PathElement {
                hashes,
                index,
                _a: Default::default(),
                _h: Default::default(),
            }]
        } else {
            Vec::new()
        };

        assert!(opts.is_empty());

        AuthPath {
            base: SubPath { path: base },
            sub: SubPath { path: sub },
            top: SubPath { path: top },
            _t: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct SubPath<H: Hasher, Arity: 'static + PoseidonArity> {
    path: Vec<PathElement<H, Arity>>,
}

#[derive(Debug, Clone)]
struct PathElement<H: Hasher, Arity: 'static + PoseidonArity> {
    hashes: Vec<Option<Fr>>,
    index: Option<usize>,
    _a: PhantomData<Arity>,
    _h: PhantomData<H>,
}

impl<H: Hasher, Arity: 'static + PoseidonArity> SubPath<H, Arity> {
    fn synthesize<CS: ConstraintSystem<Bls12>>(
        self,
        mut cs: CS,
        mut cur: num::AllocatedNum<Bls12>,
    ) -> Result<num::AllocatedNum<Bls12>, SynthesisError> {
        let arity = Arity::to_usize();

        if arity == 0 {
            // Nothing to do here.
            assert!(self.path.is_empty());
            return Ok(cur);
        }

        assert_eq!(1, arity.count_ones(), "arity must be a power of two");
        let index_bit_count = arity.trailing_zeros() as usize;

        let mut auth_path_bits = Vec::with_capacity(self.path.len());

        for (i, path) in self.path.into_iter().enumerate() {
            let elements = path.hashes;
            let indexes = path.index;

            let cs = &mut cs.namespace(|| format!("merkle tree hash {}", i));

            let mut index_bits = Vec::with_capacity(index_bit_count);

            for i in 0..index_bit_count {
                let bit = AllocatedBit::alloc(cs.namespace(|| format!("index bit {}", i)), {
                    indexes.map(|index| ((index >> i) & 1) == 1)
                })?;

                index_bits.push(Boolean::from(bit));
            }

            auth_path_bits.extend_from_slice(&index_bits);

            // Witness the authentication path elements adjacent at this depth.
            let path_elements = elements
                .iter()
                .enumerate()
                .map(|(i, elt)| {
                    num::AllocatedNum::alloc(cs.namespace(|| format!("path element {}", i)), || {
                        elt.ok_or_else(|| SynthesisError::AssignmentMissing)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let inserted = insert(cs, &cur, &index_bits, &path_elements)?;

            // Compute the new subtree value
            cur = H::Function::hash_multi_leaf_circuit::<Arity, _>(
                cs.namespace(|| "computation of commitment hash"),
                &inserted,
                i,
            )?;
        }

        // allocate input for is_right auth_path
        multipack::pack_into_inputs(cs.namespace(|| "path"), &auth_path_bits)?;

        Ok(cur)
    }
}

impl<Tree: 'static + MerkleTreeTrait> AuthPath<Tree> {
    fn blank(public_params: &<PoR<Tree> as ProofScheme<'_>>::PublicParams) -> Self {
        let base = vec![
            PathElement::<Tree::Hasher, Tree::Arity> {
                hashes: vec![None; Tree::Arity::to_usize() - 1],
                index: None,
                _a: Default::default(),
                _h: Default::default(),
            };
            graph_height::<Tree::Arity>(public_params.leaves) - 1
        ];

        let sub = if Tree::SubTreeArity::to_usize() > 0 {
            vec![PathElement::<Tree::Hasher, Tree::SubTreeArity> {
                hashes: vec![None; Tree::SubTreeArity::to_usize() - 1],
                index: None,
                _a: Default::default(),
                _h: Default::default(),
            }]
        } else {
            Vec::new()
        };

        let top = if Tree::TopTreeArity::to_usize() > 0 {
            vec![PathElement::<Tree::Hasher, Tree::TopTreeArity> {
                hashes: vec![None; Tree::TopTreeArity::to_usize() - 1],
                index: None,
                _a: Default::default(),
                _h: Default::default(),
            }]
        } else {
            Vec::new()
        };

        AuthPath {
            base: SubPath { path: base },
            sub: SubPath { path: sub },
            top: SubPath { path: top },
            _t: Default::default(),
        }
    }
}

impl<Tree: MerkleTreeTrait> CircuitComponent for PoRCircuit<Tree> {
    type ComponentPrivateInputs = Option<Root<Bls12>>;
}

pub struct PoRCompound<Tree: MerkleTreeTrait> {
    _tree: PhantomData<Tree>,
}

pub fn challenge_into_auth_path_bits<U: typenum::Unsigned>(
    challenge: usize,
    leaves: usize,
) -> Vec<bool> {
    let height = graph_height::<U>(leaves);

    let mut bits = Vec::new();
    let mut n = challenge;
    let arity = U::to_usize();
    assert_eq!(1, arity.count_ones());
    let log_arity = arity.trailing_zeros() as usize;

    for _ in 0..height - 1 {
        // Calculate the index
        let index = n % arity;
        n /= arity;

        // turn the index into bits
        for i in 0..log_arity {
            bits.push((index >> i) & 1 == 1);
        }
    }
    bits
}

impl<C: Circuit<Bls12>, P: ParameterSetMetadata, Tree: MerkleTreeTrait> CacheableParameters<C, P>
    for PoRCompound<Tree>
{
    fn cache_prefix() -> String {
        format!("proof-of-retrievability-{}", Tree::display())
    }
}

// can only implment for Bls12 because por is not generic over the engine.
impl<'a, Tree: 'static + MerkleTreeTrait> CompoundProof<'a, PoR<Tree>, PoRCircuit<Tree>>
    for PoRCompound<Tree>
{
    fn circuit<'b>(
        public_inputs: &<PoR<Tree> as ProofScheme<'a>>::PublicInputs,
        _component_private_inputs: <PoRCircuit<Tree> as CircuitComponent>::ComponentPrivateInputs,
        proof: &'b <PoR<Tree> as ProofScheme<'a>>::Proof,
        public_params: &'b <PoR<Tree> as ProofScheme<'a>>::PublicParams,
        _partition_k: Option<usize>,
    ) -> Result<PoRCircuit<Tree>> {
        let (root, private) = match (*public_inputs).commitment {
            None => (Root::Val(Some(proof.proof.root().into())), true),
            Some(commitment) => (Root::Val(Some(commitment.into())), false),
        };

        ensure!(
            private == public_params.private,
            "Inputs must be consistent with public params"
        );

        Ok(PoRCircuit::<Tree> {
            value: Root::Val(Some(proof.data.into())),
            auth_path: proof.proof.as_options().into(),
            root,
            private,
            _tree: PhantomData,
        })
    }

    fn blank_circuit(
        public_params: &<PoR<Tree> as ProofScheme<'a>>::PublicParams,
    ) -> PoRCircuit<Tree> {
        PoRCircuit::<Tree> {
            value: Root::Val(None),
            auth_path: AuthPath::blank(public_params),
            root: Root::Val(None),
            private: public_params.private,
            _tree: PhantomData,
        }
    }

    fn generate_public_inputs(
        pub_inputs: &<PoR<Tree> as ProofScheme<'a>>::PublicInputs,
        pub_params: &<PoR<Tree> as ProofScheme<'a>>::PublicParams,
        _k: Option<usize>,
    ) -> Result<Vec<Fr>> {
        let mut inputs = Vec::new();

        let get_challenge_index = |challenge: usize, arity: usize, height: usize| {
            let n = challenge;
            assert_eq!(1, arity.count_ones());

            let mut leaves = 1;
            for _ in 0..height {
                leaves *= arity;
            }
            let (index, reduced_challenge) = (challenge % leaves, challenge / leaves);
            (index, reduced_challenge)
        };

        let height = compound_path_length::<Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>(
            pub_params.leaves,
        );
        if Tree::TopTreeArity::to_usize() > 0 {
            let top_leaves = Tree::TopTreeArity::to_usize();
            let sub_leaves = Tree::SubTreeArity::to_usize();
            let base_leaves = pub_params.leaves / top_leaves / sub_leaves;

            let (base_challenge, reduced_challenge) =
                get_challenge_index(pub_inputs.challenge, Tree::Arity::to_usize(), height - 2);
            let (sub_challenge, reduced_challenge) =
                get_challenge_index(reduced_challenge, Tree::SubTreeArity::to_usize(), 1);
            let (top_challenge, _) =
                get_challenge_index(reduced_challenge, Tree::TopTreeArity::to_usize(), 1);

            {
                let base_bits =
                    challenge_into_auth_path_bits::<Tree::Arity>(base_challenge, base_leaves);
                let base_packed = multipack::compute_multipacking::<Bls12>(&base_bits);
                inputs.extend(base_packed);
            }

            {
                let sub_bits =
                    challenge_into_auth_path_bits::<Tree::SubTreeArity>(sub_challenge, sub_leaves);
                let sub_packed = multipack::compute_multipacking::<Bls12>(&sub_bits);
                inputs.extend(sub_packed);
            }

            {
                let top_bits =
                    challenge_into_auth_path_bits::<Tree::TopTreeArity>(top_challenge, top_leaves);
                let top_packed = multipack::compute_multipacking::<Bls12>(&top_bits);
                inputs.extend(top_packed);
            }
        } else if Tree::SubTreeArity::to_usize() > 0 {
            let sub_leaves = Tree::SubTreeArity::to_usize();
            let base_leaves = pub_params.leaves / sub_leaves;

            let (base_challenge, reduced_challenge) =
                get_challenge_index(pub_inputs.challenge, Tree::Arity::to_usize(), height - 1);
            let (sub_challenge, _) =
                get_challenge_index(reduced_challenge, Tree::SubTreeArity::to_usize(), 1);

            {
                let base_bits =
                    challenge_into_auth_path_bits::<Tree::Arity>(base_challenge, base_leaves);
                let base_packed = multipack::compute_multipacking::<Bls12>(&base_bits);
                inputs.extend(base_packed);
            }

            {
                let sub_bits =
                    challenge_into_auth_path_bits::<Tree::SubTreeArity>(sub_challenge, sub_leaves);
                let sub_packed = multipack::compute_multipacking::<Bls12>(&sub_bits);
                inputs.extend(sub_packed);
            }
        } else {
            let base_challenge = pub_inputs.challenge;
            let base_leaves = pub_params.leaves;

            {
                let base_bits =
                    challenge_into_auth_path_bits::<Tree::Arity>(base_challenge, base_leaves);
                let base_packed = multipack::compute_multipacking::<Bls12>(&base_bits);
                inputs.extend(base_packed);
            }
        }

        if let Some(commitment) = pub_inputs.commitment {
            ensure!(!pub_params.private, "Params must be public");
            inputs.push(commitment.into());
        } else {
            ensure!(pub_params.private, "Params must be private");
        }

        Ok(inputs)
    }
}

impl<'a, Tree: MerkleTreeTrait> Circuit<Bls12> for PoRCircuit<Tree> {
    /// # Public Inputs
    ///
    /// This circuit expects the following public inputs.
    ///
    /// * [0] - packed version of the `is_right` components of the auth_path.
    /// * [1] - the merkle root of the tree.
    ///
    /// This circuit derives the following private inputs from its fields:
    /// * value_num - packed version of `value` as bits. (might be more than one Fr)
    ///
    /// Note: All public inputs must be provided as `E::Fr`.
    fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let value = self.value;
        let auth_path = self.auth_path;
        let root = self.root;

        let base_arity = Tree::Arity::to_usize();
        let sub_arity = Tree::SubTreeArity::to_usize();
        let top_arity = Tree::TopTreeArity::to_usize();

        // All arities must be powers of two or circuits cannot be generated.
        assert_eq!(
            1,
            base_arity.count_ones(),
            "base arity must be power of two"
        );
        if sub_arity > 0 {
            assert_eq!(
                1,
                sub_arity.count_ones(),
                "subtree arity must be power of two"
            );
        }
        if top_arity > 0 {
            assert_eq!(
                1,
                top_arity.count_ones(),
                "top tree arity must be power of two"
            );
        }

        {
            let value_num = value.allocated(cs.namespace(|| "value"))?;
            let cur = value_num;

            // Ascend the merkle tree authentication path

            // base tree
            let cur = auth_path.base.synthesize(cs.namespace(|| "base"), cur)?;

            // sub
            let cur = auth_path.sub.synthesize(cs.namespace(|| "sub"), cur)?;

            // top
            let cur = auth_path.top.synthesize(cs.namespace(|| "top"), cur)?;

            {
                // Validate that the root of the merkle tree that we calculated is the same as the input.
                let rt = root.allocated(cs.namespace(|| "root_value"))?;
                constraint::equal(cs, || "enforce root is correct", &cur, &rt);

                if !self.private {
                    // Expose the root
                    rt.inputize(cs.namespace(|| "root"))?;
                }
            }

            Ok(())
        }
    }
}

impl<'a, Tree: MerkleTreeTrait> PoRCircuit<Tree> {
    #[allow(clippy::type_complexity)]
    pub fn synthesize<CS>(
        mut cs: CS,
        value: Root<Bls12>,
        auth_path: Vec<(Vec<Option<Fr>>, Option<usize>)>,
        root: Root<Bls12>,
        private: bool,
    ) -> Result<(), SynthesisError>
    where
        CS: ConstraintSystem<Bls12>,
    {
        let por = Self {
            value,
            auth_path: auth_path.into(),
            root,
            private,
            _tree: PhantomData,
        };

        por.synthesize(&mut cs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::proof::NoRequirements;
    use bellperson::gadgets::multipack;
    use ff::Field;
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    use merkletree::merkle::{is_merkle_tree_size_valid, FromIndexedParallelIterator, MerkleTree};
    use merkletree::store::VecStore;

    use crate::compound_proof;
    use crate::drgraph::{new_seed, BucketGraph, Graph, BASE_DEGREE};
    use crate::fr32::{bytes_into_fr, fr_into_bytes};
    use crate::gadgets::{MetricCS, TestConstraintSystem};
    use crate::hasher::{
        Blake2sHasher, Domain, Hasher, PedersenHasher, PoseidonArity, PoseidonDomain,
        PoseidonHasher, Sha256Hasher,
    };
    use crate::merkle::{BinaryMerkleTree, DiskStore, MerkleProofTrait, MerkleTreeWrapper};
    use crate::por;
    use crate::proof::ProofScheme;
    use crate::util::data_at_node;

    type TestTree<H, A> =
        MerkleTreeWrapper<H, VecStore<<H as Hasher>::Domain>, A, typenum::U0, typenum::U0>;

    type TestTree2<H, A, B> =
        MerkleTreeWrapper<H, VecStore<<H as Hasher>::Domain>, A, B, typenum::U0>;

    type TestTree3<H, A, B, C> = MerkleTreeWrapper<H, VecStore<<H as Hasher>::Domain>, A, B, C>;

    type ResTree<Tree> = MerkleTreeWrapper<
        <Tree as MerkleTreeTrait>::Hasher,
        <Tree as MerkleTreeTrait>::Store,
        <Tree as MerkleTreeTrait>::Arity,
        <Tree as MerkleTreeTrait>::SubTreeArity,
        <Tree as MerkleTreeTrait>::TopTreeArity,
    >;

    #[allow(clippy::type_complexity)]
    fn generate_base_tree<R: Rng, Tree: MerkleTreeTrait>(
        rng: &mut R,
        nodes: usize,
    ) -> (Vec<u8>, ResTree<Tree>) {
        let elements = (0..nodes)
            .map(|_| <Tree::Hasher as Hasher>::Domain::random(rng))
            .collect::<Vec<_>>();
        let mut data = Vec::new();
        for el in &elements {
            data.extend_from_slice(AsRef::<[u8]>::as_ref(el));
        }
        (
            data,
            MerkleTreeWrapper::try_from_iter(elements.iter().map(|v| Ok(*v))).unwrap(),
        )
    }

    #[allow(clippy::type_complexity)]
    fn generate_sub_tree<R: Rng, Tree: MerkleTreeTrait>(
        rng: &mut R,
        nodes: usize,
    ) -> (Vec<u8>, ResTree<Tree>) {
        let base_tree_count = Tree::SubTreeArity::to_usize();
        let base_tree_size = nodes / base_tree_count;
        let mut trees = Vec::with_capacity(base_tree_count);
        let mut data = Vec::new();

        for _ in 0..base_tree_count {
            let (inner_data, tree) = generate_base_tree::<
                R,
                MerkleTreeWrapper<Tree::Hasher, Tree::Store, Tree::Arity>,
            >(rng, base_tree_size);
            trees.push(tree);
            data.extend(inner_data);
        }
        (data, MerkleTreeWrapper::from_trees(trees).unwrap())
    }

    fn generate_tree<Tree: MerkleTreeTrait, R: Rng>(
        rng: &mut R,
        nodes: usize,
    ) -> (Vec<u8>, ResTree<Tree>) {
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
                >(rng, nodes / top_tree_arity);

                sub_trees.push(tree);
                data.extend(inner_data);
            }
            (data, MerkleTreeWrapper::from_sub_trees(sub_trees).unwrap())
        } else {
            if sub_tree_arity > 0 {
                generate_sub_tree::<R, Tree>(rng, nodes)
            } else {
                generate_base_tree::<R, Tree>(rng, nodes)
            }
        }
    }

    #[test]
    #[ignore] // Slow test – run only when compiled for release.
    fn por_test_compound() {
        let rng = &mut XorShiftRng::from_seed(crate::TEST_SEED);
        let leaves = 64; // good for 2, 4 and 8

        let data: Vec<u8> = (0..leaves)
            .flat_map(|_| fr_into_bytes(&Fr::random(rng)))
            .collect();
        let graph = BucketGraph::<PedersenHasher>::new(leaves, BASE_DEGREE, 0, new_seed()).unwrap();
        let tree = graph
            .merkle_tree::<BinaryMerkleTree<PedersenHasher>>(None, data.as_slice())
            .unwrap();

        let public_inputs = por::PublicInputs {
            challenge: 2,
            commitment: Some(tree.root()),
        };

        let setup_params = compound_proof::SetupParams {
            vanilla_params: por::SetupParams {
                leaves,
                private: false,
            },
            partitions: None,
            priority: false,
        };
        let public_params = PoRCompound::<BinaryMerkleTree<PedersenHasher>>::setup(&setup_params)
            .expect("setup failed");

        let private_inputs = por::PrivateInputs::<BinaryMerkleTree<PedersenHasher>>::new(
            bytes_into_fr(data_at_node(data.as_slice(), public_inputs.challenge).unwrap())
                .expect("failed to create Fr from node data")
                .into(),
            &tree,
        );

        let gparams = PoRCompound::<BinaryMerkleTree<PedersenHasher>>::groth_params(
            Some(rng),
            &public_params.vanilla_params,
        )
        .expect("failed to generate groth params");

        let proof = PoRCompound::<BinaryMerkleTree<PedersenHasher>>::prove(
            &public_params,
            &public_inputs,
            &private_inputs,
            &gparams,
        )
        .expect("failed while proving");

        let verified = PoRCompound::<BinaryMerkleTree<PedersenHasher>>::verify(
            &public_params,
            &public_inputs,
            &proof,
            &NoRequirements,
        )
        .expect("failed while verifying");
        assert!(verified);

        let (circuit, inputs) = PoRCompound::<BinaryMerkleTree<PedersenHasher>>::circuit_for_test(
            &public_params,
            &public_inputs,
            &private_inputs,
        )
        .unwrap();

        let mut cs = TestConstraintSystem::new();

        circuit.synthesize(&mut cs).expect("failed to synthesize");
        assert!(cs.is_satisfied());
        assert!(cs.verify(&inputs));
    }

    #[test]
    fn test_por_circuit_pedersen_base_2() {
        test_por_circuit::<TestTree<PedersenHasher, typenum::U2>>(3, 8_247);
    }

    #[test]
    fn test_por_circuit_blake2s_base_2() {
        test_por_circuit::<TestTree<Blake2sHasher, typenum::U2>>(3, 129_135);
    }

    #[test]
    fn test_por_circuit_sha256_base_2() {
        test_por_circuit::<TestTree<Sha256Hasher, typenum::U2>>(3, 272_295);
    }

    #[test]
    fn test_por_circuit_poseidon_base_2() {
        test_por_circuit::<TestTree<PoseidonHasher, typenum::U2>>(3, 1_905);
    }

    #[test]
    fn test_por_circuit_pedersen_base_4() {
        test_por_circuit::<TestTree<PedersenHasher, typenum::U4>>(3, 12_411);
    }

    #[test]
    fn test_por_circuit_pedersen_sub_8_2() {
        test_por_circuit::<TestTree2<PedersenHasher, typenum::U8, typenum::U2>>(4, 1_455);
    }

    #[test]
    fn test_por_circuit_pedersen_top_8_4_2() {
        test_por_circuit::<TestTree3<PedersenHasher, typenum::U8, typenum::U4, typenum::U2>>(
            5, 24_869,
        );
    }

    #[test]
    fn test_por_circuit_pedersen_top_8_2_4() {
        // We can handle top-heavy trees with a non-zero subtree arity.
        // These should never be produced, though.
        test_por_circuit::<TestTree3<PedersenHasher, typenum::U8, typenum::U2, typenum::U4>>(
            5, 24_869,
        );
    }

    #[test]
    fn test_por_circuit_blake2s_base_4() {
        test_por_circuit::<TestTree<Blake2sHasher, typenum::U4>>(3, 130_308);
    }

    #[test]
    fn test_por_circuit_sha256_base_4() {
        test_por_circuit::<TestTree<Sha256Hasher, typenum::U4>>(3, 216_270);
    }

    #[test]
    fn test_por_circuit_poseidon_base_4() {
        test_por_circuit::<TestTree<PoseidonHasher, typenum::U4>>(3, 1_185);
    }

    #[test]
    fn test_por_circuit_pedersen_base_8() {
        test_por_circuit::<TestTree<PedersenHasher, typenum::U8>>(3, 19_357);
    }

    #[test]
    fn test_por_circuit_blake2s_base_8() {
        test_por_circuit::<TestTree<Blake2sHasher, typenum::U8>>(3, 174_571);
    }

    #[test]
    fn test_por_circuit_sha256_base_8() {
        test_por_circuit::<TestTree<Sha256Hasher, typenum::U8>>(3, 251_055);
    }

    #[test]
    fn test_por_circuit_poseidon_base_8() {
        test_por_circuit::<TestTree<PoseidonHasher, typenum::U8>>(3, 1_137);
    }

    #[test]
    fn test_por_circuit_poseidon_sub_8_2() {
        test_por_circuit::<TestTree2<PoseidonHasher, typenum::U8, typenum::U2>>(4, 1_455);
    }

    #[test]
    fn test_por_circuit_poseidon_top_8_4_2() {
        test_por_circuit::<TestTree3<PoseidonHasher, typenum::U8, typenum::U4, typenum::U2>>(
            5, 1_850,
        );
    }

    #[test]
    fn test_por_circuit_poseidon_top_8_8() {
        // This is the shape we want for 32GiB sectors.
        test_por_circuit::<TestTree2<PoseidonHasher, typenum::U8, typenum::U8>>(4, 1_705);
    }
    #[test]
    fn test_por_circuit_poseidon_top_8_8_2() {
        // This is the shape we want for 64GiB secotrs.
        test_por_circuit::<TestTree3<PoseidonHasher, typenum::U8, typenum::U8, typenum::U2>>(
            5, 2_023,
        );
    }

    #[test]
    fn test_por_circuit_poseidon_top_8_2_4() {
        // We can handle top-heavy trees with a non-zero subtree arity.
        // These should never be produced, though.
        test_por_circuit::<TestTree3<PoseidonHasher, typenum::U8, typenum::U2, typenum::U4>>(
            5, 1_850,
        );
    }

    fn test_por_circuit<Tree: 'static + MerkleTreeTrait>(
        num_inputs: usize,
        num_constraints: usize,
    ) {
        let rng = &mut XorShiftRng::from_seed(crate::TEST_SEED);

        let arity = Tree::Arity::to_usize();
        assert_eq!(1, arity.count_ones());

        let sub_arity = Tree::SubTreeArity::to_usize();
        if sub_arity > 0 {
            assert_eq!(1, sub_arity.count_ones());
        }

        let top_arity = Tree::TopTreeArity::to_usize();
        if top_arity > 0 {
            assert_eq!(1, top_arity.count_ones());
        }

        // Ensure arity will evenly fill tree.
        let mut leaves = 64; // good for 2, 4 and 8

        if sub_arity > 0 {
            leaves *= sub_arity;
        }

        if top_arity > 0 {
            leaves *= top_arity;
        }

        // -- Basic Setup
        let (data, tree) = generate_tree::<Tree, _>(rng, leaves);

        for i in 0..leaves {
            println!("challenge: {}, ({})", i, leaves);

            // -- PoR
            let pub_params = por::PublicParams {
                leaves,
                private: false,
            };
            let pub_inputs = por::PublicInputs::<<Tree::Hasher as Hasher>::Domain> {
                challenge: i,
                commitment: Some(tree.root()),
            };
            let leaf = data_at_node(data.as_slice(), pub_inputs.challenge).unwrap();
            let leaf_element = <Tree::Hasher as Hasher>::Domain::try_from_bytes(leaf).unwrap();
            let priv_inputs = por::PrivateInputs::<ResTree<Tree>>::new(leaf_element, &tree);
            let p = tree.gen_proof(i).unwrap();
            assert!(p.verify());

            // create a non circuit proof
            let proof = por::PoR::<ResTree<Tree>>::prove(&pub_params, &pub_inputs, &priv_inputs)
                .expect("proving failed");

            // make sure it verifies
            let is_valid = por::PoR::<ResTree<Tree>>::verify(&pub_params, &pub_inputs, &proof)
                .expect("verification failed");
            assert!(is_valid, "failed to verify por proof");

            // -- Circuit

            let mut cs = TestConstraintSystem::<Bls12>::new();
            let por = PoRCircuit::<ResTree<Tree>> {
                value: Root::Val(Some(proof.data.into())),
                auth_path: proof.proof.as_options().into(),
                root: Root::Val(Some(pub_inputs.commitment.unwrap().into())),
                private: false,
                _tree: PhantomData,
            };

            por.synthesize(&mut cs).expect("circuit synthesis failed");
            assert!(cs.is_satisfied(), "constraints not satisfied");

            assert_eq!(cs.num_inputs(), num_inputs, "wrong number of inputs");
            assert_eq!(
                cs.num_constraints(),
                num_constraints,
                "wrong number of constraints"
            );

            let generated_inputs = PoRCompound::<ResTree<Tree>>::generate_public_inputs(
                &pub_inputs,
                &pub_params,
                None,
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

            assert!(cs.verify(&generated_inputs), "failed to verify inputs");
        }
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_pedersen_base_2() {
        private_por_test_compound::<TestTree<PedersenHasher, typenum::U2>>();
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_pedersen_base_4() {
        private_por_test_compound::<TestTree<PedersenHasher, typenum::U4>>();
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_poseidon_base_2() {
        private_por_test_compound::<TestTree<PoseidonHasher, typenum::U2>>();
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_poseidon_base_4() {
        private_por_test_compound::<TestTree<PoseidonHasher, typenum::U4>>();
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_poseidon_sub_8_2() {
        private_por_test_compound::<TestTree2<PoseidonHasher, typenum::U8, typenum::U2>>();
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_poseidon_top_8_4_2() {
        private_por_test_compound::<TestTree3<PoseidonHasher, typenum::U8, typenum::U4, typenum::U2>>(
        );
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_poseidon_top_8_8() {
        private_por_test_compound::<TestTree2<PoseidonHasher, typenum::U8, typenum::U8>>();
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_poseidon_top_8_8_2() {
        private_por_test_compound::<TestTree3<PoseidonHasher, typenum::U8, typenum::U8, typenum::U2>>(
        );
    }

    #[ignore] // Slow test – run only when compiled for release.
    #[test]
    fn test_private_por_compound_poseidon_top_8_2_4() {
        private_por_test_compound::<TestTree3<PoseidonHasher, typenum::U8, typenum::U2, typenum::U4>>(
        );
    }

    fn private_por_test_compound<Tree: 'static + MerkleTreeTrait>() {
        let rng = &mut XorShiftRng::from_seed(crate::TEST_SEED);

        let arity = Tree::Arity::to_usize();
        assert_eq!(1, arity.count_ones());

        let sub_arity = Tree::SubTreeArity::to_usize();
        if sub_arity > 0 {
            assert_eq!(1, sub_arity.count_ones());
        }

        let top_arity = Tree::TopTreeArity::to_usize();
        if top_arity > 0 {
            assert_eq!(1, top_arity.count_ones());
        }

        // Ensure arity will evenly fill tree.
        let mut leaves = 64; // good for 2, 4 and 8

        if sub_arity > 0 {
            leaves *= sub_arity;
        }

        if top_arity > 0 {
            leaves *= top_arity;
        }

        // -- Basic Setup
        let (data, tree) = generate_tree::<Tree, _>(rng, leaves);

        for i in 0..3 {
            let public_inputs = por::PublicInputs {
                challenge: i,
                commitment: None,
            };

            let setup_params = compound_proof::SetupParams {
                vanilla_params: por::SetupParams {
                    leaves,
                    private: true,
                },
                partitions: None,
                priority: false,
            };
            let public_params =
                PoRCompound::<ResTree<Tree>>::setup(&setup_params).expect("setup failed");

            let private_inputs = por::PrivateInputs::<ResTree<Tree>>::new(
                bytes_into_fr(data_at_node(data.as_slice(), public_inputs.challenge).unwrap())
                    .expect("failed to create Fr from node data")
                    .into(),
                &tree,
            );

            {
                let (circuit, inputs) =
                    PoRCompound::circuit_for_test(&public_params, &public_inputs, &private_inputs)
                        .unwrap();

                let mut cs = TestConstraintSystem::new();

                circuit.synthesize(&mut cs).expect("failed to synthesize");

                if !cs.is_satisfied() {
                    panic!(
                        "failed to satisfy: {:?}",
                        cs.which_is_unsatisfied().unwrap()
                    );
                }
                assert!(
                    cs.verify(&inputs),
                    "verification failed with TestContraintSystem and generated inputs"
                );
            }
            // NOTE: This diagnostic code currently fails, even though the proof generated from the blank circuit verifies.
            // Use this to debug differences between blank and regular circuit generation.
            {
                let (circuit1, _inputs) =
                    PoRCompound::circuit_for_test(&public_params, &public_inputs, &private_inputs)
                        .unwrap();
                let blank_circuit =
                    PoRCompound::<ResTree<Tree>>::blank_circuit(&public_params.vanilla_params);

                let mut cs_blank = MetricCS::new();
                blank_circuit
                    .synthesize(&mut cs_blank)
                    .expect("failed to synthesize");

                let a = cs_blank.pretty_print_list();

                let mut cs1 = TestConstraintSystem::new();
                circuit1.synthesize(&mut cs1).expect("failed to synthesize");
                let b = cs1.pretty_print_list();

                for (i, (a, b)) in a.chunks(100).zip(b.chunks(100)).enumerate() {
                    assert_eq!(a, b, "failed at chunk {}", i);
                }
            }

            let blank_groth_params = PoRCompound::<ResTree<Tree>>::groth_params(
                Some(rng),
                &public_params.vanilla_params,
            )
            .expect("failed to generate groth params");

            let proof = PoRCompound::prove(
                &public_params,
                &public_inputs,
                &private_inputs,
                &blank_groth_params,
            )
            .expect("failed while proving");

            let verified =
                PoRCompound::verify(&public_params, &public_inputs, &proof, &NoRequirements)
                    .expect("failed while verifying");

            assert!(verified);
        }
    }

    #[test]
    fn test_private_por_input_circuit_pedersen_binary() {
        test_private_por_input_circuit::<TestTree<PedersenHasher, typenum::U2>>(8_246);
    }

    #[test]
    fn test_private_por_input_circuit_poseidon_binary() {
        test_private_por_input_circuit::<TestTree<PoseidonHasher, typenum::U2>>(1_904);
    }

    #[test]
    fn test_private_por_input_circuit_pedersen_quad() {
        test_private_por_input_circuit::<TestTree<PedersenHasher, typenum::U4>>(12_410);
    }

    #[test]
    fn test_private_por_input_circuit_poseidon_quad() {
        test_private_por_input_circuit::<TestTree<PoseidonHasher, typenum::U4>>(1_184);
    }

    fn test_private_por_input_circuit<Tree: MerkleTreeTrait>(num_constraints: usize) {
        let rng = &mut XorShiftRng::from_seed(crate::TEST_SEED);

        let leaves = 64; // good for 2, 4 and 8

        for i in 0..leaves {
            // -- Basic Setup

            let data: Vec<u8> = (0..leaves)
                .flat_map(|_| fr_into_bytes(&Fr::random(rng)))
                .collect();

            let graph =
                BucketGraph::<Tree::Hasher>::new(leaves, BASE_DEGREE, 0, new_seed()).unwrap();
            let tree = graph.merkle_tree::<Tree>(None, data.as_slice()).unwrap();

            // -- PoR

            let pub_params = por::PublicParams {
                leaves,
                private: true,
            };
            let pub_inputs = por::PublicInputs {
                challenge: i,
                commitment: None,
            };

            let priv_inputs = por::PrivateInputs::<Tree>::new(
                bytes_into_fr(data_at_node(data.as_slice(), pub_inputs.challenge).unwrap())
                    .unwrap()
                    .into(),
                &tree,
            );

            // create a non circuit proof
            let proof = por::PoR::<Tree>::prove(&pub_params, &pub_inputs, &priv_inputs)
                .expect("proving failed");

            // make sure it verifies
            let is_valid = por::PoR::<Tree>::verify(&pub_params, &pub_inputs, &proof)
                .expect("verification failed");
            assert!(is_valid, "failed to verify por proof");

            // -- Circuit

            let mut cs = TestConstraintSystem::<Bls12>::new();

            let por = PoRCircuit::<Tree> {
                value: Root::Val(Some(proof.data.into())),
                auth_path: proof.proof.as_options().into(),
                root: Root::Val(Some(tree.root().into())),
                private: true,
                _tree: PhantomData,
            };

            por.synthesize(&mut cs).expect("circuit synthesis failed");
            assert!(cs.is_satisfied(), "constraints not satisfied");

            assert_eq!(cs.num_inputs(), 2, "wrong number of inputs");
            assert_eq!(
                cs.num_constraints(),
                num_constraints,
                "wrong number of constraints"
            );

            let auth_path_bits = challenge_into_auth_path_bits::<Tree::Arity>(
                pub_inputs.challenge,
                pub_params.leaves,
            );
            let packed_auth_path = multipack::compute_multipacking::<Bls12>(&auth_path_bits);

            let mut expected_inputs = Vec::new();
            expected_inputs.extend(packed_auth_path);

            assert_eq!(cs.get_input(0, "ONE"), Fr::one(), "wrong input 0");

            assert_eq!(
                cs.get_input(1, "base/path/input 0"),
                expected_inputs[0],
                "wrong packed_auth_path"
            );

            assert!(cs.is_satisfied(), "constraints are not all satisfied");
            assert!(cs.verify(&expected_inputs), "failed to verify inputs");
        }
    }
}
