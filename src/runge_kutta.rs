// This implements a simple Runge-Kutta method, more specifically the variant
// known as DOPRI5(4), cf. https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
// supplemented by the adaptive step size algorithm outlined in the paper
// https://www.researchgate.net/publication/226932720.

use ndarray::Array1;
use std::ops::Deref;

const A: [[f64; 6]; 6] = [
    [1. / 5., 0., 0., 0., 0., 0.],
    [3. / 40., 9. / 40., 0., 0., 0., 0.],
    [44. / 45., -56. / 15., 32. / 9., 0., 0., 0.],
    [
        19372. / 6561.,
        -25360. / 2187.,
        64448. / 6561.,
        -212. / 729.,
        0.,
        0.,
    ],
    [
        9017. / 3168.,
        -355. / 33.,
        46732. / 5247.,
        49. / 176.,
        -5103. / 18656.,
        0.,
    ],
    [
        35. / 384.,
        0.,
        500. / 1113.,
        125. / 192.,
        -2187. / 6784.,
        11. / 84.,
    ],
];
const B: [f64; 7] = [
    35. / 384.,
    0.,
    500. / 1113.,
    125. / 192.,
    -2187. / 6784.,
    11. / 84.,
    0.,
];
const DELTA_B: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -686.0 / 13487.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];
const C: [f64; 6] = [0.2, 0.3, 0.8, 8. / 9., 1., 1.];
const D: [f64; 7] = [
    -12715105075.0 / 11282082432.0,
    0.0,
    87487479700.0 / 32700410799.0,
    -10690763975.0 / 1880347072.0,
    701980252875.0 / 199316789632.0,
    -1453857185.0 / 822651844.0,
    69997945.0 / 29380423.0,
];
const KAPPA: f64 = 0.5;
const ALPHA: f64 = 0.17;
const BETA: f64 = 0.04;
const MAX_FACTOR: f64 = 10.0;
const MIN_FACTOR: f64 = 0.2;
const THETA: f64 = 0.9;
const EPSILON0: f64 = 1e-10;
const EPSILON1: f64 = 1e-15;
const H0: f64 = 1e-6;
const HMAX: f64 = 0.1;

pub type Vector = Array1<f64>;
type AutonomousVectorField = Box<dyn Fn(&Vector) -> Vector>;
type NonAutonomousVectorField = Box<dyn Fn(f64, &Vector) -> Vector>;

fn copy(v: &Vector) -> Vector {
    let mut w = Vector::zeros(v.len());
    w.assign(&v);
    w
}

fn abs(v: &Vector) -> f64 {
    v.dot(v).sqrt()
}

pub struct VectorField {
    f: NonAutonomousVectorField,
}

impl VectorField {
    pub fn autonomous(f: AutonomousVectorField) -> VectorField {
        VectorField {
            f: Box::new(move |_t: f64, x: &Vector| (*f)(x)),
        }
    }
    pub fn new(f: NonAutonomousVectorField) -> VectorField {
        VectorField { f: f }
    }
}

impl Deref for VectorField {
    type Target = NonAutonomousVectorField;
    fn deref(&self) -> &Self::Target {
        &self.f
    }
}

pub struct RungeKuttaSolver {
    f: VectorField,                             // The vector field in question
    h: f64,                                     // Current stepsize
    tn: f64,                                    // Current time
    xn: Vector,                                 // Current solution vector
    k: [Vector; 7],                             // Current quadrature values
    r: [Vector; 5],                             // Coefficients for dense output
    norm: Box<dyn Fn(&Vector, &Vector)->f64>,   // Error measurment derived from user input atol and rtol
    en: f64,                                    // Last accepted error
    dim: usize,                                 // Dimension of the vector field
    reject: bool,                               // Last attempt was rejected
}

impl RungeKuttaSolver {
    pub fn new(f: VectorField, t0: f64, x0: Vector, rtol: f64, atol: f64) -> RungeKuttaSolver {
        let dim = x0.len();
        let norm = move |x: &Vector, x0: &Vector| {
            x.iter()
                .zip(x0)
                .map(|(xi, x0i)| (xi / (0.1 * atol + x0i.abs() * 0.1 * rtol)).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let h = RungeKuttaSolver::initial_stepsize(&f, t0, &x0, &norm);
        RungeKuttaSolver {
            f: f,
            tn: t0,
            xn: x0,
            h: h,
            k: [Vector::zeros(dim), Vector::zeros(dim), Vector::zeros(dim), Vector::zeros(dim),
                Vector::zeros(dim), Vector::zeros(dim), Vector::zeros(dim)],
            r: [Vector::zeros(dim), Vector::zeros(dim), Vector::zeros(dim), Vector::zeros(dim),
                Vector::zeros(dim)],
            dim: dim,
            norm: Box::new(norm),
            en: 0.,
            reject: false,
        }
    }

    pub fn time(&self) -> f64 {
        self.tn
    }

    pub fn on_mesh(&mut self, t_max: f64, stepsize: f64) -> Vec<f64> {
        let capacity = ((t_max - self.tn) / stepsize).ceil() as usize;
        let mut result = Vec::<f64>::with_capacity(capacity);
        let mut t = 0.;
        while self.tn <= t_max {
            t += stepsize;
            let mut r = self.r.clone();
            let mut h = self.h;
            let mut tn = self.tn;
            while self.tn + self.h <= t {
                r = self.r.clone();
                h = self.h;
                tn = self.tn;
                self.next();
            }
            let theta = (t - tn) / h;
            let theta1 = 1. - theta;
            let x = copy(&r[0]) + copy(&r[1]) * theta + copy(&r[2]) * theta * theta1 
                    + copy(&r[3]) * theta.powi(2) * theta1 + copy(&r[4]) * theta.powi(2) * theta1.powi(2);
            result.extend(x.iter());
        }
        result
    }

    fn initial_stepsize(f: &VectorField, t0: f64, x0: &Vector, norm: &dyn Fn(&Vector, &Vector) -> f64) -> f64 {
        let f0 = f(t0, x0);
        let d0 = norm(x0, x0);
        let d1 = norm(&f0, x0);
        let h0 = if d0 < EPSILON0 || d1 < EPSILON0 {
            H0
        } else {
            1e-2 * d0 / d1
        };
        let x1 = x0 + &(h0 * &f0);
        let f1 = f(t0 + h0, &x1);
        let d2 = norm(&(f1 - f0), x0) / h0;
        let h1 = if d1 < EPSILON1 && d2 < EPSILON1 {
            H0.max(1e-3 * h0)
        } else {
            1e-2 / (d1.max(d2)).powf(0.2)
        };
        (100.0 * h0).min(h1).min(HMAX)
    }

    fn approximate(&mut self) -> (Vector, f64) {
        let k0 = (self.f)(self.tn, &self.xn);
        self.k = [
            k0,
            Vector::zeros(self.dim),
            Vector::zeros(self.dim),
            Vector::zeros(self.dim),
            Vector::zeros(self.dim),
            Vector::zeros(self.dim),
            Vector::zeros(self.dim),
        ];
        for (j, c) in C.iter().enumerate() {
            let delta_x: Vector = A[j][0..j + 1]
                .iter()
                .zip(&self.k[0..j + 1])
                .map(|(&aji, ki)| self.h * aji * ki)
                .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
            self.k[j + 1] = (self.f)(self.tn + c * self.h, &(delta_x + &self.xn));
        }
        let xn = copy(&self.xn)
            + &B.iter()
                .zip(&self.k)
                .map(|(&bi, ki)| self.h * bi * ki)
                .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
        let eps_x = &DELTA_B
            .iter()
            .zip(&self.k)
            .map(|(&bi, ki)| self.h * bi * ki)
            .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
        let x_test = (&self.xn).iter().zip(&xn).map(|(x1i, &x2i)| x1i.abs().max(x2i)).collect();
        let eps = (*self.norm)(&eps_x, &x_test) / (self.dim as f64).sqrt();
        (xn, eps)
    }

    fn stepsize_factor(&self, eps: f64) -> f64 {
        let mut f = THETA * eps.powf(-ALPHA);
        if eps < 1. {
            f *= (self.en.powf(BETA)).max(KAPPA.powf(BETA));
            if self.reject {
                f = f.min(1.);
            }
        }
        (f.max(MIN_FACTOR)).min(MAX_FACTOR)
    }
}

impl Iterator for RungeKuttaSolver {
    type Item = Vector;
    fn next(&mut self) -> Option<Vector> {
        let (x_tmp, eps) = self.approximate();
        let old_h = self.h;
        self.h *= self.stepsize_factor(eps);
        self.h = self.h;
        self.r[0] = copy(&self.xn);
        if eps < 1. {
            self.xn = x_tmp;
            self.tn += old_h;
            self.en = eps;
            self.reject = false;
        } else {
            self.reject = true;
            let retry = self.approximate();
            self.xn = retry.0;
            self.tn += self.h;
        }
        self.r[1] = &self.xn - &self.r[0];
        self.r[2] = &self.k[0] * self.h - &self.r[1];
        self.r[3] = 2. * &self.r[1] - (&self.k[0] + &self.k[1]) * self.h;
        self.r[4] = D.iter()
                        .zip(&self.k)
                        .map(|(&di, ki)| self.h * di * ki)
                        .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
        Some(copy(&self.xn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn mirror(v: &Vector) -> Vector {
        arr2(&[[0., -1.], [1., 0.]]).dot(v)
    }

    fn squash(v: &Vector) -> Vector {
        arr2(&[[2., 0.], [0., -1.]]).dot(v)
    }

    fn rotation_solution(t: f64) -> Vector {
        arr1(&[t.cos(), t.sin()])
    }

    fn hyperbolic_solution(t: f64) -> Vector {
        arr1(&[(2.0 * t).exp(), (-t).exp()])
    }

    fn test_value_against_accurate(time: f64, solution: Vector, accurate: Vector, tolerance: f64) {
        let abs_error = abs(&(&solution - &accurate));
        let rel_error = abs(&solution) / abs(&accurate) - 1.0;
        dbg!(time, &solution, &accurate, &abs_error, &rel_error);
        assert!(abs_error < tolerance || rel_error < tolerance);
    }

    fn test_solver_against_analytic(
        vector_field: VectorField,
        analytic: Solution,
        t_min: f64,
        x0: Vector,
        t_max: f64,
        tolerance: f64,
    ) {
        let mut solution = RungeKuttaSolver::new(vector_field, t_min, x0, tolerance, tolerance);
        let mut approx;
        let mut time;
        while solution.time() < t_max {
            approx = solution.next().unwrap();
            time = solution.time();
            test_value_against_accurate(time, approx, (*analytic)(time), tolerance);
        }
    }

    #[test]
    fn rotation_solved_accurately() {
        let vector_field = VectorField::autonomous(Box::new(mirror));
        let analytic = Box::new(rotation_solution);
        test_solver_against_analytic(
            vector_field,
            analytic,
            0.,
            arr1(&[1., 0.]),
            2.0 * 3.141592653,
            1e-10,
        );
    }

    #[test]
    fn linear_diagonal_solved_accurately() {
        let vector_field = VectorField::autonomous(Box::new(squash));
        let analytic = Box::new(hyperbolic_solution);
        test_solver_against_analytic(
            vector_field,
            analytic,
            0.,
            arr1(&[1., 1.]),
            2. * 2.71828,
            1e-10,
        );
    }
}
