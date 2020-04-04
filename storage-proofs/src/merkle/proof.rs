#![allow(clippy::len_without_is_empty)]

use std::convert::TryFrom;
use std::marker::PhantomData;

use anyhow::{ensure, Error, Result};
use generic_array::typenum::{self, Unsigned, U0};
use merkletree::hash::Algorithm;
use merkletree::proof;
use paired::bls12_381::Fr;
use serde::{Deserialize, Serialize};

use crate::drgraph::graph_height;
use crate::hasher::{Domain, Hasher, PoseidonArity};

/// Trait to abstract over the concept of Merkle Proof.
pub trait MerkleProofTrait:
    Clone + Serialize + serde::de::DeserializeOwned + std::fmt::Debug + Sync + Send
{
    type Hasher: Hasher;
    type Arity: 'static + PoseidonArity;
    type SubTreeArity: 'static + PoseidonArity;
    type TopTreeArity: 'static + PoseidonArity;

    /// Try to convert a merkletree proof into this structure.
    fn try_from_proof(
        p: proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof<
    H: Hasher,
    BaseArity: PoseidonArity,
    SubTreeArity: PoseidonArity = U0,
    TopTreeArity: PoseidonArity = U0,
> {
    #[serde(bound(
        serialize = "H::Domain: Serialize",
        deserialize = "H::Domain: Deserialize<'de>"
    ))]
    data: ProofData<H, BaseArity, SubTreeArity, TopTreeArity>,
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

    fn try_from_proof(
        p: proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
    ) -> Result<Self> {
        if p.top_layer_nodes() > 0 {
            Ok(MerkleProof {
                data: ProofData::Top(TopProof::try_from_proof(p)?),
            })
        } else if p.sub_layer_nodes() > 0 {
            Ok(MerkleProof {
                data: ProofData::Sub(SubProof::try_from_proof(p)?),
            })
        } else {
            Ok(MerkleProof {
                data: ProofData::Single(SingleProof::try_from_proof(p)?),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ProofData<
    H: Hasher,
    BaseArity: PoseidonArity,
    SubTreeArity: PoseidonArity,
    TopTreeArity: PoseidonArity,
> {
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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct SingleProof<H: Hasher, Arity: PoseidonArity> {
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

impl<H: Hasher, Arity: PoseidonArity> SingleProof<H, Arity> {
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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct SubProof<H: Hasher, BaseArity: PoseidonArity, SubTreeArity: PoseidonArity> {
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

impl<H: Hasher, BaseArity: PoseidonArity, SubTreeArity: PoseidonArity>
    SubProof<H, BaseArity, SubTreeArity>
{
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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct TopProof<
    H: Hasher,
    BaseArity: PoseidonArity,
    SubTreeArity: PoseidonArity,
    TopTreeArity: PoseidonArity,
> {
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

impl<
        H: Hasher,
        BaseArity: PoseidonArity,
        SubTreeArity: PoseidonArity,
        TopTreeArity: PoseidonArity,
    > TopProof<H, BaseArity, SubTreeArity, TopTreeArity>
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
        BaseArity: PoseidonArity,
        SubTreeArity: PoseidonArity,
        TopTreeArity: PoseidonArity,
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
fn proof_to_single<H: Hasher, Arity: PoseidonArity, TargetArity: PoseidonArity>(
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

    fn try_from_proof(
        p: proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
    ) -> Result<Self> {
        Ok(proof_to_single(&p, 1, None))
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

    fn try_from_proof(
        p: proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
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
        let sub_proof = proof_to_single(&p, 0, Some(base_p.root()));

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
        for _i in 0..self.base_proof.path().len() {
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

    fn try_from_proof(
        p: proof::Proof<<Self::Hasher as Hasher>::Domain, Self::Arity>,
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
            &p,
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
        for _i in 0..self.base_proof.path().len() {
            base_proof_leaves *= BaseArity::to_usize()
        }

        let sub_proof_leaves = base_proof_leaves * SubTreeArity::to_usize();

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

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    use rand;
    use std::io::Write;

    use crate::drgraph::{new_seed, BucketGraph, Graph, BASE_DEGREE};
    use crate::hasher::{Blake2sHasher, PedersenHasher, PoseidonHasher, Sha256Hasher};
    use crate::merkle::MerkleProofTrait;

    fn merklepath<H: 'static + Hasher, BaseTreeArity: 'static + PoseidonArity>() {
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
        let mut rng = rand::thread_rng();
        let mut data = Vec::new();
        for _ in 0..leafs {
            let elt: H::Domain = H::Domain::random(&mut rng);
            data.push(elt);
        }

        let mut trees = Vec::with_capacity(SubTreeArity::to_usize());
        for _ in 0..SubTreeArity::to_usize() {
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
            let proof = SubProof::<H, BaseTreeArity, SubTreeArity>::try_from_proof(
                tree.gen_proof(i).unwrap(),
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
        let mut rng = rand::thread_rng();
        let mut data = Vec::new();
        for _ in 0..leafs {
            let elt: H::Domain = H::Domain::random(&mut rng);
            data.push(elt);
        }

        let mut sub_trees = Vec::with_capacity(TopTreeArity::to_usize());
        for _i in 0..TopTreeArity::to_usize() {
            let mut trees = Vec::with_capacity(SubTreeArity::to_usize());
            for _i in 0..SubTreeArity::to_usize() {
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
            let proof = TopProof::<H, BaseTreeArity, SubTreeArity, TopTreeArity>::try_from_proof(
                tree.gen_proof(i).unwrap(),
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
