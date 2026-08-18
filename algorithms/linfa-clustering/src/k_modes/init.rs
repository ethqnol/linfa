use std::collections::{HashMap, HashSet};
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand::Rng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// Specifies the centroid initialization strategy for K-Modes.
pub enum KModesInit<T> {
    /// Density and dissimilarity-based initialization (Cao et al. [2009]). Default.
    Cao,
    /// Density-based initialization (Huang [1997, 1998]).
    Huang,
    /// Randomly selects `n_clusters` unique observations from the dataset.
    Random,
    /// Precomputed initial centroids with shape `(n_clusters, n_features)`.
    Precomputed(Array2<T>),
}

impl<T> Default for KModesInit<T> {
    fn default() -> Self {
        Self::Cao
    }
}

impl<T: crate::k_modes::EquivalenceTarget> KModesInit<T> {
    /// Runs the chosen initialization routine
    pub(crate) fn run<R: Rng>(
        &self,
        n_clusters: usize,
        observations: ArrayView2<T>,
        rng: &mut R,
    ) -> Array2<T> {
        match self {
            Self::Cao => init_cao(observations, n_clusters),
            Self::Huang => init_huang(observations, n_clusters, rng),
            Self::Random => random_init(observations, n_clusters, rng),
            Self::Precomputed(centroids) => {
                assert_eq!(
                    centroids.nrows(),
                    n_clusters,
                    "Precomputed centroids must have shape (n_clusters, n_features)"
                );
                assert_eq!(
                    centroids.ncols(),
                    observations.ncols(),
                    "Precomputed centroids must match feature count"
                );
                centroids.clone()
            }
        }
    }
}

/// Cao initialization (Cao et al. [2009]): Density and dissimilarity-based initialization.
pub(crate) fn init_cao<T: crate::k_modes::EquivalenceTarget>(
    x: ArrayView2<T>,
    n_clusters: usize,
) -> Array2<T> {
    let (n_points, n_attrs) = x.dim();
    assert!(
        n_clusters <= n_points,
        "n_clusters cannot exceed number of data points"
    );

    // Calculate point density across categorical features
    // dens[i] = sum_{j=0}^{n_attrs-1} freq_j(x[i, j]) / (n_points * n_attrs)
    let mut dens = vec![0.0f64; n_points];
    for iattr in 0..n_attrs {
        let mut freq: HashMap<&T, usize> = HashMap::new();
        for ipoint in 0..n_points {
            *freq.entry(&x[[ipoint, iattr]]).or_insert(0) += 1;
        }
        let denom = (n_points * n_attrs) as f64;
        for ipoint in 0..n_points {
            let count = freq.get(&x[[ipoint, iattr]]).copied().unwrap_or(0);
            dens[ipoint] += (count as f64) / denom;
        }
    }

    let mut selected_indices = Vec::with_capacity(n_clusters);
    // Centroid 0: point with highest density
    let first_idx = dens
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    selected_indices.push(first_idx);

    // Remaining centroids: max of (min distance * density) to existing centroids
    for _ in 1..n_clusters {
        let mut best_idx = 0;
        let mut best_score = -1.0f64;

        for ipoint in 0..n_points {
            if selected_indices.contains(&ipoint) {
                continue;
            }

            let row = x.row(ipoint);
            let mut min_d = f64::INFINITY;

            for &c_idx in &selected_indices {
                let c_row = x.row(c_idx);
                let dist = row
                    .iter()
                    .zip(c_row.iter())
                    .filter(|(a, b)| a != b)
                    .count() as f64;
                let score = dist * dens[ipoint];
                if score < min_d {
                    min_d = score;
                }
            }

            if min_d > best_score {
                best_score = min_d;
                best_idx = ipoint;
            }
        }
        selected_indices.push(best_idx);
    }

    Array2::from_shape_fn((n_clusters, n_attrs), |(i, j)| {
        x[[selected_indices[i], j]].clone()
    })
}

/// Huang initialization (Huang [1997, 1998]): Attribute frequency sampling.
pub(crate) fn init_huang<T: crate::k_modes::EquivalenceTarget, R: Rng>(
    x: ArrayView2<T>,
    n_clusters: usize,
    rng: &mut R,
) -> Array2<T> {
    let (n_points, n_attrs) = x.dim();
    assert!(
        n_clusters <= n_points,
        "n_clusters cannot exceed number of data points"
    );

    // Sample tentative centroids using the frequency distribution of attributes
    let mut tentative = Array2::from_shape_fn((n_clusters, n_attrs), |(_, j)| {
        let rand_idx = rng.gen_range(0..n_points);
        x[[rand_idx, j]].clone()
    });

    // Set each centroid to the closest unique point in X
    let mut selected_indices = HashSet::new();
    for ik in 0..n_clusters {
        let tent_row = tentative.row(ik);
        let mut best_idx = 0;
        let mut min_dist = usize::MAX;

        for ipoint in 0..n_points {
            if selected_indices.contains(&ipoint) {
                continue;
            }

            let row = x.row(ipoint);
            let dist = row
                .iter()
                .zip(tent_row.iter())
                .filter(|(a, b)| a != b)
                .count();

            if dist < min_dist {
                min_dist = dist;
                best_idx = ipoint;
            }
        }

        selected_indices.insert(best_idx);
        for j in 0..n_attrs {
            tentative[[ik, j]] = x[[best_idx, j]].clone();
        }
    }

    tentative
}

/// Random initialization: selects `n_clusters` random unique data points as centroids.
pub(crate) fn random_init<T: Clone, R: Rng>(
    x: ArrayView2<T>,
    n_clusters: usize,
    rng: &mut R,
) -> Array2<T> {
    let (n_points, n_attrs) = x.dim();
    assert!(
        n_clusters <= n_points,
        "n_clusters cannot exceed number of data points"
    );

    let mut selected_indices = Vec::with_capacity(n_clusters);
    let mut seen = HashSet::with_capacity(n_clusters);

    while selected_indices.len() < n_clusters {
        let idx = rng.gen_range(0..n_points);
        if seen.insert(idx) {
            selected_indices.push(idx);
        }
    }

    Array2::from_shape_fn((n_clusters, n_attrs), |(i, j)| {
        x[[selected_indices[i], j]].clone()
    })
}
