use std::collections::HashMap;

use crate::k_modes::init::KModesInit;
use crate::k_modes::{KModesError, KModesParams, KModesValidParams};
use linfa::traits::{Fit, PredictInplace};
use linfa::DatasetBase;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
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
/// A fitted K-Modes clustering model containing the cluster modes (centroids).
pub struct KModes<T: EquivalenceTarget> {
    modes: Array2<T>,
    cost: usize,
}

/// Trait bound for categorical elements supported by K-Modes.
pub trait EquivalenceTarget: PartialEq + Eq + Clone + std::hash::Hash {}
impl<T: PartialEq + Eq + Clone + std::hash::Hash> EquivalenceTarget for T {}

impl<T: EquivalenceTarget> KModes<T> {
    pub fn params(n_clusters: usize) -> KModesParams<T, Xoshiro256Plus> {
        KModesParams::new(n_clusters)
    }

    pub fn params_with_rng<R: Rng>(n_clusters: usize, rng: R) -> KModesParams<T, R> {
        KModesParams::new_with_rng(n_clusters, rng)
    }

    /// Cluster modes matrix with shape `(n_clusters, n_features)`.
    pub fn modes(&self) -> &Array2<T> {
        &self.modes
    }

    /// Total dissimilarity cost of the fitted clustering.
    pub fn cost(&self) -> usize {
        self.cost
    }
}

impl<T: EquivalenceTarget, R: Rng + Clone, D: Data<Elem = T>, L>
    Fit<ArrayBase<D, Ix2>, L, KModesError> for KModesValidParams<T, R>
{
    type Object = KModes<T>;

    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, L>,
    ) -> Result<Self::Object, KModesError> {
        let observations = dataset.records().view();
        let (n_points, _) = observations.dim();

        if n_points == 0 {
            return Err(
                linfa::error::Error::Parameters("Dataset cannot be empty".to_string()).into(),
            );
        }

        if self.n_clusters() > n_points {
            return Err(linfa::error::Error::Parameters(format!(
                "Cannot have more clusters ({}) than data points ({})",
                self.n_clusters(),
                n_points
            ))
            .into());
        }

        let mut rng = self.rng().clone();
        let mut best_cost = usize::MAX;
        let mut best_centroids = None;
        let mut best_run = 0;

        let n_runs = match self.init_method() {
            KModesInit::Cao if self.n_runs() > 1 => {
                if self.verbose() {
                    println!("Cao initialization is deterministic. Running 1 initialization.");
                }
                1
            }
            _ => self.n_runs(),
        };

        for run_idx in 0..n_runs {
            if self.verbose() && n_runs > 1 {
                println!("Starting K-Modes run {}/{}", run_idx + 1, n_runs);
            }

            let (centroids, cost) = k_modes_single(
                observations,
                self.n_clusters(),
                self.max_n_iterations(),
                self.init_method(),
                &mut rng,
                self.verbose(),
            );

            if cost < best_cost {
                best_cost = cost;
                best_centroids = Some(centroids);
                best_run = run_idx;
            }
        }

        if self.verbose() && n_runs > 1 {
            println!("Best run was number {} (cost: {})", best_run + 1, best_cost);
        }

        let modes = best_centroids.ok_or_else(|| {
            linfa::error::Error::Parameters("Failed to fit K-Modes centroids".to_string())
        })?;

        Ok(KModes {
            modes,
            cost: best_cost,
        })
    }
}

/// Simple matching dissimilarity between two records.
pub(crate) fn matching_dissim<T: PartialEq>(a: ArrayView1<T>, b: ArrayView1<T>) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Finds the nearest centroid index and its distance.
pub(crate) fn closest_centroid<T: PartialEq>(
    centroids: ArrayView2<T>,
    observation: ArrayView1<T>,
) -> (usize, usize) {
    let mut min_dist = usize::MAX;
    let mut closest_idx = 0;

    for (idx, centroid) in centroids.rows().into_iter().enumerate() {
        let dist = matching_dissim(centroid, observation);
        if dist < min_dist {
            min_dist = dist;
            closest_idx = idx;
        }
    }

    (closest_idx, min_dist)
}

fn calculate_cost<T: PartialEq>(
    observations: ArrayView2<T>,
    centroids: ArrayView2<T>,
    memberships: &[usize],
) -> usize {
    observations
        .rows()
        .into_iter()
        .zip(memberships.iter())
        .map(|(obs, &cluster_id)| matching_dissim(obs, centroids.row(cluster_id)))
        .sum()
}

fn k_modes_single<T: EquivalenceTarget, R: Rng>(
    observations: ArrayView2<T>,
    n_clusters: usize,
    max_n_iterations: u64,
    init: &KModesInit<T>,
    rng: &mut R,
    verbose: bool,
) -> (Array2<T>, usize) {
    let (n_points, n_attrs) = observations.dim();

    let mut centroids = init.run(n_clusters, observations, rng);
    let mut memberships = vec![0usize; n_points];
    let mut cluster_counts = vec![0usize; n_clusters];

    let mut cl_attr_freq: Vec<Vec<HashMap<T, usize>>> = (0..n_clusters)
        .map(|_| (0..n_attrs).map(|_| HashMap::new()).collect())
        .collect();

    for (ipoint, curpoint) in observations.rows().into_iter().enumerate() {
        let (clust, _) = closest_centroid(centroids.view(), curpoint);
        memberships[ipoint] = clust;
        cluster_counts[clust] += 1;

        for (iattr, curattr) in curpoint.iter().enumerate() {
            *cl_attr_freq[clust][iattr]
                .entry(curattr.clone())
                .or_insert(0) += 1;
        }
    }

    for ik in 0..n_clusters {
        if cluster_counts[ik] == 0 {
            let random_idx = rng.gen_range(0..n_points);
            let sample_row = observations.row(random_idx);
            for iattr in 0..n_attrs {
                centroids[[ik, iattr]] = sample_row[iattr].clone();
            }
        } else {
            for iattr in 0..n_attrs {
                let mode_val = cl_attr_freq[ik][iattr]
                    .iter()
                    .filter(|(_, &count)| count > 0)
                    .max_by_key(|(_, &count)| count)
                    .map(|(k, _)| k.clone())
                    .unwrap_or_else(|| observations[[0, iattr]].clone());
                centroids[[ik, iattr]] = mode_val;
            }
        }
    }

    let mut cost = calculate_cost(observations, centroids.view(), &memberships);

    let mut iter = 0;
    while iter < max_n_iterations {
        iter += 1;
        let mut moves = 0;

        for ipoint in 0..n_points {
            let curpoint = observations.row(ipoint);
            let (to_clust, _) = closest_centroid(centroids.view(), curpoint);
            let from_clust = memberships[ipoint];

            if to_clust == from_clust {
                continue;
            }

            moves += 1;
            memberships[ipoint] = to_clust;
            cluster_counts[from_clust] -= 1;
            cluster_counts[to_clust] += 1;

            move_point_cat(
                curpoint,
                to_clust,
                from_clust,
                &mut cl_attr_freq,
                &mut centroids,
            );

            // Reassign a point from the largest cluster if a cluster became empty
            if cluster_counts[from_clust] == 0 {
                let largest_clust = cluster_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &count)| count)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let candidates: Vec<usize> = memberships
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &c)| if c == largest_clust { Some(idx) } else { None })
                    .collect();

                if !candidates.is_empty() {
                    let rindx = candidates[rng.gen_range(0..candidates.len())];
                    let rpoint = observations.row(rindx);

                    memberships[rindx] = from_clust;
                    cluster_counts[largest_clust] -= 1;
                    cluster_counts[from_clust] += 1;

                    move_point_cat(
                        rpoint,
                        from_clust,
                        largest_clust,
                        &mut cl_attr_freq,
                        &mut centroids,
                    );
                }
            }
        }

        let ncost = calculate_cost(observations, centroids.view(), &memberships);
        cost = ncost;

        if moves == 0 {
            if verbose {
                println!(
                    "K-Modes converged at iteration {} (moves: {}, cost: {})",
                    iter, moves, cost
                );
            }
            break;
        } else if verbose && (iter % 10 == 0 || iter == max_n_iterations) {
            println!(
                "Iteration {}/{}: moves = {}, cost = {}",
                iter, max_n_iterations, moves, cost
            );
        }
    }

    (centroids, cost)
}

fn move_point_cat<T: EquivalenceTarget>(
    point: ArrayView1<T>,
    to_clust: usize,
    from_clust: usize,
    cl_attr_freq: &mut [Vec<HashMap<T, usize>>],
    centroids: &mut Array2<T>,
) {
    for (iattr, curattr) in point.iter().enumerate() {
        let to_map = &mut cl_attr_freq[to_clust][iattr];
        let to_count = to_map.entry(curattr.clone()).or_insert(0);
        *to_count += 1;
        let new_val_freq = *to_count;

        let current_centroid_val = &centroids[[to_clust, iattr]];
        let current_centroid_freq = to_map.get(current_centroid_val).copied().unwrap_or(0);
        if new_val_freq > current_centroid_freq {
            centroids[[to_clust, iattr]] = curattr.clone();
        }

        let from_map = &mut cl_attr_freq[from_clust][iattr];
        if let Some(from_count) = from_map.get_mut(curattr) {
            if *from_count > 0 {
                *from_count -= 1;
            }
        }

        let old_centroid_val = &centroids[[from_clust, iattr]];
        if old_centroid_val == curattr {
            if let Some((best_val, _)) = from_map
                .iter()
                .filter(|(_, &count)| count > 0)
                .max_by_key(|(_, &count)| count)
            {
                centroids[[from_clust, iattr]] = best_val.clone();
            }
        }
    }
}

impl<T: EquivalenceTarget, D: Data<Elem = T>> PredictInplace<ArrayBase<D, Ix2>, Array1<usize>>
    for KModes<T>
{
    fn predict_inplace(&self, observations: &ArrayBase<D, Ix2>, memberships: &mut Array1<usize>) {
        assert_eq!(
            observations.nrows(),
            memberships.len(),
            "Number of observations must match memberships length"
        );

        for (i, obs) in observations.rows().into_iter().enumerate() {
            memberships[i] = closest_centroid(self.modes.view(), obs).0;
        }
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<usize> {
        Array1::zeros(x.nrows())
    }
}

impl<T: EquivalenceTarget, D: Data<Elem = T>> PredictInplace<ArrayBase<D, Ix1>, usize>
    for KModes<T>
{
    fn predict_inplace(&self, observation: &ArrayBase<D, Ix1>, membership: &mut usize) {
        *membership = closest_centroid(self.modes.view(), observation.view()).0;
    }

    fn default_target(&self, _x: &ArrayBase<D, Ix1>) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::traits::Predict;
    use ndarray::array;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<KModes<usize>>();
    }

    #[test]
    fn test_kmodes_simple_fitting() {
        let data = array![
            ["A", "X"],
            ["A", "X"],
            ["B", "Y"],
            ["B", "Y"],
            ["A", "X"],
            ["B", "Y"]
        ];

        let dataset = DatasetBase::from(data);
        let model = KModes::params(2)
            .max_n_iterations(50)
            .verbose(false)
            .fit(&dataset)
            .unwrap();

        assert_eq!(model.modes().dim(), (2, 2));

        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 6);

        assert_eq!(predictions[0], predictions[1]);
        assert_eq!(predictions[0], predictions[4]);

        assert_eq!(predictions[2], predictions[3]);
        assert_eq!(predictions[2], predictions[5]);
        assert_ne!(predictions[0], predictions[2]);
    }

    #[test]
    fn test_kmodes_single_observation_predict() {
        let data = array![["A", "X"], ["B", "Y"]];
        let dataset = DatasetBase::from(data);
        let model = KModes::params(2).fit(&dataset).unwrap();

        let single_sample = array!["A", "X"];
        let pred: usize = model.predict(&single_sample);
        assert!(pred < 2);
    }
}
