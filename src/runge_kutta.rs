// This implements a simple Runge-Kutta-Fehlberg method. 
// Cf. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method.

use ndarray::Array1;
pub type Vector = Array1<f64>;

pub enum VectorField {
    Autonomous(Box<dyn Fn(Vector) -> Vector>),
    NonAutonomous(Box<dyn Fn(f64, Vector) -> Vector>)
}

const A: [[f64; 5]; 5] = [[0.25, 0., 0., 0., 0.], 
                           [3./32., 9./32., 0., 0., 0.],
                           [1932./2197., -7200./2197., 7296./2197., 0., 0.],
                           [439./216., -8., 3680./513., -845./4104., 0.],
                           [-8./27., 2., -3544./2565., 1859./4104., -0.275]];
const B: [f64; 6] = [16./135., 0., 6656./12825., 28561./56430., -0.18, 2./55.];
const DELTA_B: [f64; 6] = [16./135. - 25./216. , 0., 6656./12825. - 1408./2565., 28561./56430. - 2197./4104., 0.02,	2./55.];
const C: [f64; 5] = [0.25, 0.375, 12./13., 1., 0.5];

pub struct RungeKuttaSolver {
    f: Box<dyn Fn(f64, Vector) -> Vector>,  // The vector field in question
    h: f64,                                 // Current stepsize
    tn: f64,                                // Current time
    xn: Vector,                             // Current solution vector
    tol: f64,                               // Local error tolerance
    dim: usize                              // Dimension of the vector field
}

impl RungeKuttaSolver {
    pub fn new(vector_field: VectorField, t0: f64, x0: Vector, step_size: f64, tolerance: f64) -> RungeKuttaSolver {
        let dim = x0.len();
        let f = match vector_field { VectorField::Autonomous(vector_field) => { Box::new(move |_t: f64, x: Vector| (*vector_field)(x)) },
                                     VectorField::NonAutonomous(vector_field) => vector_field };
        RungeKuttaSolver { f: f, tn: t0, xn: x0, h: step_size, dim: dim, tol: tolerance }
    }
    pub fn time(&self) -> f64 { self.tn }        
    fn approximate(&self) -> (Vector, f64) {
        let mut x_tmp = Vector::zeros(self.dim);
        x_tmp.assign(&self.xn);
        let k0 = (*self.f)(self.tn, x_tmp);
        let mut k = [k0, Vector::zeros(self.dim), Vector::zeros(self.dim), Vector::zeros(self.dim), 
                     Vector::zeros(self.dim), Vector::zeros(self.dim), Vector::zeros(self.dim)];
        for (j, c) in C.iter().enumerate() {
            let delta_x: Vector = A[j][0 .. j+1].iter().zip(&k[0 .. j+1])
                                    .map(|(&aji, ki)| self.h * aji * ki)
                                    .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
            let mut x_tmp = Vector::zeros(self.dim);
            x_tmp.assign(&self.xn);
            k[j + 1] = (*self.f)(self.tn + c * self.h, delta_x + x_tmp);
        }
        let mut x_tmp = Vector::zeros(self.dim);
        x_tmp.assign(&self.xn);
        let xn = x_tmp + &B.iter().zip(&k)
                            .map(|(&bi, ki)| self.h * bi * ki)
                            .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
        let eps = &DELTA_B.iter().zip(&k)
                    .map(|(&bi, ki)| self.h * bi * ki)
                    .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
        let err = eps.dot(eps).sqrt();
        println!("{}, {}, {}", err, self.tol, self.h);
        (xn, err)
    }
}

impl Iterator for RungeKuttaSolver {
    type Item = Vector;
    fn next(&mut self) -> Option<Vector> {
        let (x_tmp, eps) = self.approximate();
        let old_h = self.h;
        self.h *= 0.9 * (self.tol / eps).powf( if eps < self.tol { 0.2 } else { 0.25 });
        if eps < self.tol {
            self.xn = x_tmp;
            self.tn += old_h;
        } 
        else {
            self.xn = self.approximate().0;
            self.tn += self.h;
        }
        let mut result = Vector::zeros(self.dim);
        result.assign(&self.xn);
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ arr1, arr2 };

    fn rotate(v: Vector) -> Vector {
        arr2(&[[0., -1.], [1., 0.]]).dot(&v)
    }

    fn squash(v: Vector) -> Vector {
        arr2(&[[2., 0.], [0., -1.]]).dot(&v)
    }

    fn test_value_against_accurate(time: f64, solution: Vector, accurate: Vector, tolerance: f64) {
        println!("t={} approx: {} accurate: {}", time, solution, accurate);
        let abs_err = solution - accurate;
        let error_small = (abs_err).dot(&abs_err).sqrt() < tolerance;
        assert!(error_small);
    }

    fn test_solver_against_analytic() {
        
    }

    #[test]
    fn rotation_solved_accurately() {
        let rotation = VectorField::Autonomous(Box::new(rotate));
        let mut solution = RungeKuttaSolver::new(rotation, 0., arr1(&[1., 0.]), 1e-2, 1e-13);
        let mut approx;
        let mut time;
        while solution.time() < 2. * 3.141592653 {
            approx = solution.next().unwrap();
            time = solution.time();
            test_value_against_accurate(time, approx[0], time.cos(), 1e-12);
            test_value_against_accurate(time, approx[1], time.sin(), 1e-12);
        }
    }

    #[test]
    fn linear_diagonal_solved_accurately() {
        let linear = VectorField::Autonomous(Box::new(squash));
        let mut solution = RungeKuttaSolver::new(linear, 0., arr1(&[1., 1.]), 1e-2, 1e-10);
        let mut time;
        let mut approx;
        let mut steps_taken = 0;
        while solution.time() < 5.0 {
            approx = solution.next().unwrap();
            time = solution.time();
            test_value_against_accurate(time, approx[0], (2.0 * time).exp(), 1e-6);
            test_value_against_accurate(time, approx[1], (-time).exp(), 1e-6);
            steps_taken += 1;
        }
        println!("Steps taken: {}", steps_taken);
        // assert!(steps_taken <= 500);
    }
}