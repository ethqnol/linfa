use crate::KModesParamsError;
use linfa::ParamGuard;
use ndarray_rand::rand::Rng;
use rand_xoshiro::Xoshiro256Plus;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
pub struct KModesValidParams<R: Rng> {
    pub(crate) n_clusters: usize,
    pub(crate) max_n_iterations: u64,
    pub(crate) rng: R,
}

impl<R: Rng> KModesValidParams<R> {
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn max_n_iterations(&self) -> u64 {
        self.max_n_iterations
    }

    pub fn rng(&self) -> &R {
        &self.rng
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
pub struct KModesParams<R: Rng>(KModesValidParams<R>);

impl KModesParams<Xoshiro256Plus> {
    pub fn new(n_clusters: usize) -> Self {
        Self::new_with_rng(n_clusters, Xoshiro256Plus::seed_from_u64(42))
    }
}

impl<R: Rng> KModesParams<R> {
    pub fn new_with_rng(n_clusters: usize, rng: R) -> Self {
        Self(KModesValidParams {
            n_clusters,
            max_n_iterations: 100,
            rng,
        })
    }

    pub fn max_n_iterations(mut self, max_n_iterations: u64) -> Self {
        self.0.max_n_iterations = max_n_iterations;
        self
    }

    pub fn with_rng<R2: Rng>(self, rng: R2) -> KModesParams<R2> {
        KModesParams(KModesValidParams {
            n_clusters: self.0.n_clusters,
            max_n_iterations: self.0.max_n_iterations,
            rng,
        })
    }
}

impl<R: Rng> ParamGuard for KModesParams<R> {
    type Checked = KModesValidParams<R>;
    type Error = KModesParamsError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.n_clusters == 0 {
            Err(KModesParamsError::NClusters)
        } else if self.0.max_n_iterations == 0 {
            Err(KModesParamsError::MaxIterations)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::ParamGuard;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<KModesParamsError>();
        has_autotraits::<KModesParams<Xoshiro256Plus>>();
        has_autotraits::<KModesValidParams<Xoshiro256Plus>>();
    }

    #[test]
    fn invalid_params() {
        assert_eq!(
            KModesParams::new(0).check(),
            Err(KModesParamsError::NClusters)
        );
        assert_eq!(
            KModesParams::new(2).max_n_iterations(0).check(),
            Err(KModesParamsError::MaxIterations)
        );
    }
}
