use crate::k_modes::{KModesError, KModesParams, KModesValidParams};
use linfa::traits::{Fit, PredictInplace};
use linfa::DatasetBase;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use ndarray_rand::rand::Rng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KModes<T: EquivalenceTarget> {
    modes: Array2<T>,
}

pub trait EquivalenceTarget: PartialEq + Clone {}
impl<T: PartialEq + Clone> EquivalenceTarget for T {}

impl KModes<usize> {
    pub fn params(n_clusters: usize) -> KModesParams<Xoshiro256Plus> {
        KModesParams::new(n_clusters)
    }

    pub fn params_with_rng<R: Rng>(n_clusters: usize, rng: R) -> KModesParams<R> {
        KModesParams::new_with_rng(n_clusters, rng)
    }
}

impl<T: EquivalenceTarget> KModes<T> {
    pub fn modes(&self) -> &Array2<T> {
        &self.modes
    }
}

impl<T: EquivalenceTarget, R: Rng + Clone, D: Data<Elem = T>, L>
    Fit<ArrayBase<D, Ix2>, L, KModesError> for KModesValidParams<R>
{
    type Object = KModes<T>;

    fn fit(
        &self,
        _dataset: &DatasetBase<ArrayBase<D, Ix2>, L>,
    ) -> Result<Self::Object, KModesError> {
        todo!("Implement K-Modes fitting algorithm logic")
    }
}

impl<T: EquivalenceTarget, D: Data<Elem = T>> PredictInplace<ArrayBase<D, Ix2>, Array1<usize>>
    for KModes<T>
{
    fn predict_inplace(&self, _observations: &ArrayBase<D, Ix2>, _memberships: &mut Array1<usize>) {
        todo!("Implement K-Modes prediction logic")
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<usize> {
        Array1::zeros(x.nrows())
    }
}

impl<T: EquivalenceTarget, D: Data<Elem = T>> PredictInplace<ArrayBase<D, Ix1>, usize>
    for KModes<T>
{
    fn predict_inplace(&self, _observation: &ArrayBase<D, Ix1>, _membership: &mut usize) {
        todo!("Implement single observation K-Modes prediction logic")
    }

    fn default_target(&self, _x: &ArrayBase<D, Ix1>) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<KModes<usize>>();
    }
}
