use crate::distribution::{Discrete, DiscreteCDF};
use crate::function::{beta, factorial};
use crate::statistics::*;
use std::f64;

/// Implements the
/// [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Binomial, Discrete};
/// use statrs::statistics::Distribution;
///
/// let n = Binomial::new(0.5, 5).unwrap();
/// assert_eq!(n.mean().unwrap(), 2.5);
/// assert_eq!(n.pmf(0), 0.03125);
/// assert_eq!(n.pmf(3), 0.3125);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Binomial {
    p: f64,
    n: u64,
}

/// Represents the errors that can occur when creating a [`Binomial`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum BinomialError {
    /// The probability is NaN or not in `[0, 1]`.
    ProbabilityInvalid,
}

impl std::fmt::Display for BinomialError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BinomialError::ProbabilityInvalid => write!(f, "Probability is NaN or not in [0, 1]"),
        }
    }
}

impl std::error::Error for BinomialError {}

impl Binomial {
    /// Constructs a new binomial distribution
    /// with a given `p` probability of success of `n`
    /// trials.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is `NaN`, less than `0.0`,
    /// greater than `1.0`, or if `n` is less than `0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Binomial;
    ///
    /// let mut result = Binomial::new(0.5, 5);
    /// assert!(result.is_ok());
    ///
    /// result = Binomial::new(-0.5, 5);
    /// assert!(result.is_err());
    /// ```
    pub fn new(p: f64, n: u64) -> Result<Binomial, BinomialError> {
        if p.is_nan() || !(0.0..=1.0).contains(&p) {
            Err(BinomialError::ProbabilityInvalid)
        } else {
            Ok(Binomial { p, n })
        }
    }

    /// Returns the probability of success `p` of
    /// the binomial distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Binomial;
    ///
    /// let n = Binomial::new(0.5, 5).unwrap();
    /// assert_eq!(n.p(), 0.5);
    /// ```
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Returns the number of trials `n` of the
    /// binomial distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Binomial;
    ///
    /// let n = Binomial::new(0.5, 5).unwrap();
    /// assert_eq!(n.n(), 5);
    /// ```
    pub fn n(&self) -> u64 {
        self.n
    }
}

impl std::fmt::Display for Binomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bin({},{})", self.p, self.n)
    }
}

#[allow(non_snake_case)]
fn btpe<R: rand::Rng>(n: u64, p: f64, rng: &mut R) -> Option<u32> {
    use rand::prelude::Distribution;

    let (n, n_int) = match u32::try_from(n) {
        Ok(n_32) => (f64::from(n_32), n_32),
        // more than u32::MAX: not supported
        Err(_) => return None,
    };

    if !p.is_finite() {
        return None;
    }

    // step 0
    let r = p.min(1.0 - p);

    // is this needed?
    assert!(n * r >= 10.0);

    let q = 1.0 - r;
    let f_M = n * r + r;

    // could maybe benefit from unsafe:
    // SAFETY:
    // p is in [0, 1].
    // -> r is in [0, 0.5].
    // -> f_M is in [0, 0.5 * n + 0.5].
    // n is in [0, u32::MAX], so f_M < u32::MAX
    // let M: u32 = unsafe { f_M.to_int_unchecked() };
    let M = f_M as u32;

    // r * q is in [0, 0.25].
    let nrq = n * r * q;
    let p1 = (2.195 * nrq.sqrt() - 4.6 * q).floor() + 0.5;
    let x_M = M as f64 + 0.5;
    let x_L = x_M - p1;
    let x_R = x_M + p1;
    let c = 0.134 + 20.5 / (15.3 + M as f64);
    let a = (f_M - x_L) / (f_M - x_L * r);
    let lambda_L = a * (1.0 + a / 2.0);
    let a = (x_R - f_M) / (x_R * q);
    let lambda_R = a * (1.0 + a / 2.0);

    let p2 = p1 * (1.0 + 2.0 * c);
    let p3 = p2 + c / lambda_L;
    let p4 = p3 + c / lambda_R;

    use crate::distribution::Uniform;
    let u_dist = Uniform::new(0.0, p4).unwrap();
    let v_dist = Uniform::new(0.0, 1.0).unwrap();

    // this loop contains steps 1 to 5.3
    // the `y` returned here is used in step 6
    let y = loop {
        // step 1
        let u = u_dist.sample(rng);
        let v = v_dist.sample(rng);

        if u <= p1 {
            break (x_M - p1 * v + u) as u32;
        }

        let (y, v) = if u <= p2 {
            // step 2
            let x = x_L + (u - p1) / c;
            let v = v * c + 1.0 - (f_M.floor() - x + 0.5).abs() / p1;
            if v > 1.0 {
                // back to step 1
                continue;
            }

            (x as u32, v)
        } else if u <= p3 {
            // step 3
            let y = (x_L + v.ln() / lambda_L) as i32;

            match u32::try_from(y) {
                // y is negative -> back to step 1
                Err(_) => continue,
                Ok(y) => (y, v * (u - p2) * lambda_L),
            }
        } else {
            // step 4
            let y = (x_R - v.ln() / lambda_R) as u32;
            if y > n_int {
                // back to step 1
                continue;
            }

            (y, v)
        };

        let v = v * (u - p3) * lambda_R;

        // step 5.0
        let k = y.abs_diff(M) as i32;
        // note: the paper uses k < (nrq) / 2 - 1, we can remove the -1 because of `as i32`.
        if k > 20 && k < (nrq / 2.0) as i32 {
            // step 5.2
            let k = k as f64;
            let rho = (k / nrq) * ((k * (k / 3.0 + 0.625) + 1.0 / 6.0) / nrq + 0.5);
            let t = -(k * k) / (2.0 * nrq);
            let A = v.ln();

            if A < t - rho {
                break y;
            }

            if A > t + rho {
                // go back to step 1
                continue;
            }

            // step 5.3
            let x1 = (y + 1) as f64;
            let f1 = (M + 1) as f64;
            let z = (n_int + 1 - M) as f64;
            let w = (n_int - y + 1) as f64;

            // Stirling's approximation to ln(f(y))
            fn estimate_ln(x: f64) -> f64 {
                let x = x * x;

                (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x) / x) / x) / x) / x / 166320.0
            }

            if A > x_M * (f1 / x1).ln()
                + (n - M as f64 + 0.5) * (z / w)
                + (y - M) as f64 * (w * r / (x1 * q)).ln()
                + estimate_ln(f1)
                + estimate_ln(z)
                - estimate_ln(x1)
                - estimate_ln(w)
            {
                continue;
            }

            break y;
        }

        // step 5.1
        let s = r / q;
        let a = s * (n + 1.0);

        let steps = M.abs_diff(y);
        let start = M.min(y);

        let f = (start..start + steps).fold(1.0, |f, i| f / (a / i as f64 - s));

        if v <= f {
            break y;
        }

        // else: back to step 1
    };

    if p > 0.5 {
        Some(n_int - y)
    } else {
        Some(y)
    }
}

#[cfg(feature = "rand")]
impl ::rand::distributions::Distribution<f64> for Binomial {
    fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        (0..self.n).fold(0.0, |acc, _| {
            let n: f64 = rng.gen();
            if n < self.p {
                acc + 1.0
            } else {
                acc
            }
        })
    }
}

impl DiscreteCDF<u64, f64> for Binomial {
    /// Calculates the cumulative distribution function for the
    /// binomial distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// I_(1 - p)(n - x, 1 + x)
    /// ```
    ///
    /// where `I_(x)(a, b)` is the regularized incomplete beta function
    fn cdf(&self, x: u64) -> f64 {
        if x >= self.n {
            1.0
        } else {
            let k = x;
            beta::beta_reg((self.n - k) as f64, k as f64 + 1.0, 1.0 - self.p)
        }
    }

    /// Calculates the survival function for the
    /// binomial distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// I_(p)(x + 1, n - x)
    /// ```
    ///
    /// where `I_(x)(a, b)` is the regularized incomplete beta function
    fn sf(&self, x: u64) -> f64 {
        if x >= self.n {
            0.0
        } else {
            let k = x;
            beta::beta_reg(k as f64 + 1.0, (self.n - k) as f64, self.p)
        }
    }
}

impl Min<u64> for Binomial {
    /// Returns the minimum value in the domain of the
    /// binomial distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    fn min(&self) -> u64 {
        0
    }
}

impl Max<u64> for Binomial {
    /// Returns the maximum value in the domain of the
    /// binomial distribution representable by a 64-bit
    /// integer
    ///
    /// # Formula
    ///
    /// ```text
    /// n
    /// ```
    fn max(&self) -> u64 {
        self.n
    }
}

impl Distribution<f64> for Binomial {
    /// Returns the mean of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// p * n
    /// ```
    fn mean(&self) -> Option<f64> {
        Some(self.p * self.n as f64)
    }

    /// Returns the variance of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// n * p * (1 - p)
    /// ```
    fn variance(&self) -> Option<f64> {
        Some(self.p * (1.0 - self.p) * self.n as f64)
    }

    /// Returns the entropy of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / 2) * ln (2 * π * e * n * p * (1 - p))
    /// ```
    fn entropy(&self) -> Option<f64> {
        let entr = if self.p == 0.0 || ulps_eq!(self.p, 1.0) {
            0.0
        } else {
            (0..self.n + 1).fold(0.0, |acc, x| {
                let p = self.pmf(x);
                acc - p * p.ln()
            })
        };
        Some(entr)
    }

    /// Returns the skewness of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 - 2p) / sqrt(n * p * (1 - p)))
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some((1.0 - 2.0 * self.p) / (self.n as f64 * self.p * (1.0 - self.p)).sqrt())
    }
}

impl Median<f64> for Binomial {
    /// Returns the median of the binomial distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// floor(n * p)
    /// ```
    fn median(&self) -> f64 {
        (self.p * self.n as f64).floor()
    }
}

impl Mode<Option<u64>> for Binomial {
    /// Returns the mode for the binomial distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// floor((n + 1) * p)
    /// ```
    fn mode(&self) -> Option<u64> {
        let mode = if self.p == 0.0 {
            0
        } else if ulps_eq!(self.p, 1.0) {
            self.n
        } else {
            ((self.n as f64 + 1.0) * self.p).floor() as u64
        };
        Some(mode)
    }
}

impl Discrete<u64, f64> for Binomial {
    /// Calculates the probability mass function for the binomial
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (n choose k) * p^k * (1 - p)^(n - k)
    /// ```
    fn pmf(&self, x: u64) -> f64 {
        if x > self.n {
            0.0
        } else if self.p == 0.0 {
            if x == 0 {
                1.0
            } else {
                0.0
            }
        } else if ulps_eq!(self.p, 1.0) {
            if x == self.n {
                1.0
            } else {
                0.0
            }
        } else {
            (factorial::ln_binomial(self.n, x)
                + x as f64 * self.p.ln()
                + (self.n - x) as f64 * (1.0 - self.p).ln())
            .exp()
        }
    }

    /// Calculates the log probability mass function for the binomial
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln((n choose k) * p^k * (1 - p)^(n - k))
    /// ```
    fn ln_pmf(&self, x: u64) -> f64 {
        if x > self.n {
            f64::NEG_INFINITY
        } else if self.p == 0.0 {
            if x == 0 {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else if ulps_eq!(self.p, 1.0) {
            if x == self.n {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            factorial::ln_binomial(self.n, x)
                + x as f64 * self.p.ln()
                + (self.n - x) as f64 * (1.0 - self.p).ln()
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(p: f64, n: u64; Binomial; BinomialError);

    #[test]
    fn test_create() {
        create_ok(0.0, 4);
        create_ok(0.3, 3);
        create_ok(1.0, 2);
    }

    #[test]
    fn test_bad_create() {
        create_err(f64::NAN, 1);
        create_err(-1.0, 1);
        create_err(2.0, 1);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Binomial| x.mean().unwrap();
        test_exact(0.0, 4, 0.0, mean);
        test_absolute(0.3, 3, 0.9, 1e-15, mean);
        test_exact(1.0, 2, 2.0, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Binomial| x.variance().unwrap();
        test_exact(0.0, 4, 0.0, variance);
        test_exact(0.3, 3, 0.63, variance);
        test_exact(1.0, 2, 0.0, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Binomial| x.entropy().unwrap();
        test_exact(0.0, 4, 0.0, entropy);
        test_absolute(0.3, 3, 1.1404671643037712668976423399228972051669206536461, 1e-15, entropy);
        test_exact(1.0, 2, 0.0, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Binomial| x.skewness().unwrap();
        test_exact(0.0, 4, f64::INFINITY, skewness);
        test_exact(0.3, 3, 0.503952630678969636286, skewness);
        test_exact(1.0, 2, f64::NEG_INFINITY, skewness);
    }

    #[test]
    fn test_median() {
        let median = |x: Binomial| x.median();
        test_exact(0.0, 4, 0.0, median);
        test_exact(0.3, 3, 0.0, median);
        test_exact(1.0, 2, 2.0, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Binomial| x.mode().unwrap();
        test_exact(0.0, 4, 0, mode);
        test_exact(0.3, 3, 1, mode);
        test_exact(1.0, 2, 2, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Binomial| x.min();
        let max = |x: Binomial| x.max();
        test_exact(0.3, 10, 0, min);
        test_exact(0.3, 10, 10, max);
    }

    #[test]
    fn test_pmf() {
        let pmf = |arg: u64| move |x: Binomial| x.pmf(arg);
        test_exact(0.0, 1, 1.0, pmf(0));
        test_exact(0.0, 1, 0.0, pmf(1));
        test_exact(0.0, 3, 1.0, pmf(0));
        test_exact(0.0, 3, 0.0, pmf(1));
        test_exact(0.0, 3, 0.0, pmf(3));
        test_exact(0.0, 10, 1.0, pmf(0));
        test_exact(0.0, 10, 0.0, pmf(1));
        test_exact(0.0, 10, 0.0, pmf(10));
        test_exact(0.3, 1, 0.69999999999999995559107901499373838305473327636719, pmf(0));
        test_exact(0.3, 1, 0.2999999999999999888977697537484345957636833190918, pmf(1));
        test_exact(0.3, 3, 0.34299999999999993471888615204079956461021032657166, pmf(0));
        test_absolute(0.3, 3, 0.44099999999999992772448109690231306411849135972008, 1e-15, pmf(1));
        test_absolute(0.3, 3, 0.026999999999999997002397833512077451789759292859569, 1e-16, pmf(3));
        test_absolute(0.3, 10, 0.02824752489999998207939855277004937778546385011091, 1e-17, pmf(0));
        test_absolute(0.3, 10, 0.12106082099999992639752977030555903089040470780077, 1e-15, pmf(1));
        test_absolute(0.3, 10, 0.0000059048999999999978147480206303047454017251032868501, 1e-20, pmf(10));
        test_exact(1.0, 1, 0.0, pmf(0));
        test_exact(1.0, 1, 1.0, pmf(1));
        test_exact(1.0, 3, 0.0, pmf(0));
        test_exact(1.0, 3, 0.0, pmf(1));
        test_exact(1.0, 3, 1.0, pmf(3));
        test_exact(1.0, 10, 0.0, pmf(0));
        test_exact(1.0, 10, 0.0, pmf(1));
        test_exact(1.0, 10, 1.0, pmf(10));
    }

    #[test]
    fn test_ln_pmf() {
        let ln_pmf = |arg: u64| move |x: Binomial| x.ln_pmf(arg);
        test_exact(0.0, 1, 0.0, ln_pmf(0));
        test_exact(0.0, 1, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(0.0, 3, 0.0, ln_pmf(0));
        test_exact(0.0, 3, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(0.0, 3, f64::NEG_INFINITY, ln_pmf(3));
        test_exact(0.0, 10, 0.0, ln_pmf(0));
        test_exact(0.0, 10, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(0.0, 10, f64::NEG_INFINITY, ln_pmf(10));
        test_exact(0.3, 1, -0.3566749439387324423539544041072745145718090708995, ln_pmf(0));
        test_exact(0.3, 1, -1.2039728043259360296301803719337238685164245381839, ln_pmf(1));
        test_exact(0.3, 3, -1.0700248318161973270618632123218235437154272126985, ln_pmf(0));
        test_absolute(0.3, 3, -0.81871040353529122294284394322574719301255212216016, 1e-15, ln_pmf(1));
        test_absolute(0.3, 3, -3.6119184129778080888905411158011716055492736145517, 1e-15, ln_pmf(3));
        test_exact(0.3, 10, -3.566749439387324423539544041072745145718090708995, ln_pmf(0));
        test_absolute(0.3, 10, -2.1114622067804823267977785542148302920616046876506, 1e-14, ln_pmf(1));
        test_exact(0.3, 10, -12.039728043259360296301803719337238685164245381839, ln_pmf(10));
        test_exact(1.0, 1, f64::NEG_INFINITY, ln_pmf(0));
        test_exact(1.0, 1, 0.0, ln_pmf(1));
        test_exact(1.0, 3, f64::NEG_INFINITY, ln_pmf(0));
        test_exact(1.0, 3, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(1.0, 3, 0.0, ln_pmf(3));
        test_exact(1.0, 10, f64::NEG_INFINITY, ln_pmf(0));
        test_exact(1.0, 10, f64::NEG_INFINITY, ln_pmf(1));
        test_exact(1.0, 10, 0.0, ln_pmf(10));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: u64| move |x: Binomial| x.cdf(arg);
        test_exact(0.0, 1, 1.0, cdf(0));
        test_exact(0.0, 1, 1.0, cdf(1));
        test_exact(0.0, 3, 1.0, cdf(0));
        test_exact(0.0, 3, 1.0, cdf(1));
        test_exact(0.0, 3, 1.0, cdf(3));
        test_exact(0.0, 10, 1.0, cdf(0));
        test_exact(0.0, 10, 1.0, cdf(1));
        test_exact(0.0, 10, 1.0, cdf(10));
        test_absolute(0.3, 1, 0.7, 1e-15, cdf(0));
        test_exact(0.3, 1, 1.0, cdf(1));
        test_absolute(0.3, 3, 0.343, 1e-14, cdf(0));
        test_absolute(0.3, 3, 0.784, 1e-15, cdf(1));
        test_exact(0.3, 3, 1.0, cdf(3));
        test_absolute(0.3, 10, 0.0282475249, 1e-16, cdf(0));
        test_absolute(0.3, 10, 0.1493083459, 1e-14, cdf(1));
        test_exact(0.3, 10, 1.0, cdf(10));
        test_exact(1.0, 1, 0.0, cdf(0));
        test_exact(1.0, 1, 1.0, cdf(1));
        test_exact(1.0, 3, 0.0, cdf(0));
        test_exact(1.0, 3, 0.0, cdf(1));
        test_exact(1.0, 3, 1.0, cdf(3));
        test_exact(1.0, 10, 0.0, cdf(0));
        test_exact(1.0, 10, 0.0, cdf(1));
        test_exact(1.0, 10, 1.0, cdf(10));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: u64| move |x: Binomial| x.sf(arg);
        test_exact(0.0, 1, 0.0, sf(0));
        test_exact(0.0, 1, 0.0, sf(1));
        test_exact(0.0, 3, 0.0, sf(0));
        test_exact(0.0, 3, 0.0, sf(1));
        test_exact(0.0, 3, 0.0, sf(3));
        test_exact(0.0, 10, 0.0, sf(0));
        test_exact(0.0, 10, 0.0, sf(1));
        test_exact(0.0, 10, 0.0, sf(10));
        test_absolute(0.3, 1, 0.3, 1e-15, sf(0));
        test_exact(0.3, 1, 0.0, sf(1));
        test_absolute(0.3, 3, 0.657, 1e-14, sf(0));
        test_absolute(0.3, 3, 0.216, 1e-15, sf(1));
        test_exact(0.3, 3, 0.0, sf(3));
        test_absolute(0.3, 10, 0.9717524751000001, 1e-16, sf(0));
        test_absolute(0.3, 10, 0.850691654100002, 1e-14, sf(1));
        test_exact(0.3, 10, 0.0, sf(10));
        test_exact(1.0, 1, 1.0, sf(0));
        test_exact(1.0, 1, 0.0, sf(1));
        test_exact(1.0, 3, 1.0, sf(0));
        test_exact(1.0, 3, 1.0, sf(1));
        test_exact(1.0, 3, 0.0, sf(3));
        test_exact(1.0, 10, 1.0, sf(0));
        test_exact(1.0, 10, 1.0, sf(1));
        test_exact(1.0, 10, 0.0, sf(10));
    }

    #[test]
    fn test_cdf_upper_bound() {
        let cdf = |arg: u64| move |x: Binomial| x.cdf(arg);
        test_exact(0.5, 3, 1.0, cdf(5));
    }

    #[test]
    fn test_sf_upper_bound() {
        let sf = |arg: u64| move |x: Binomial| x.sf(arg);
        test_exact(0.5, 3, 0.0, sf(5));
    }

    #[test]
    fn test_inverse_cdf() {
        let invcdf = |arg: f64| move |x: Binomial| x.inverse_cdf(arg);
        test_exact(0.4, 5, 2, invcdf(0.3456));

        // cases in issue #185
        test_exact(0.018, 465, 1, invcdf(3.472e-4));
        test_exact(0.5, 6, 4, invcdf(0.75));
    }

    #[test]
    fn test_cdf_inverse_cdf() {
        let cdf_invcdf = |arg: u64| move |x: Binomial| x.inverse_cdf(x.cdf(arg));
        test_exact(0.3, 10, 3, cdf_invcdf(3));
        test_exact(0.3, 10, 4, cdf_invcdf(4));
        test_exact(0.5, 6, 4, cdf_invcdf(4));
    }

    #[test]
    fn test_discrete() {
        test::check_discrete_distribution(&create_ok(0.3, 5), 5);
        test::check_discrete_distribution(&create_ok(0.7, 10), 10);
    }
}
