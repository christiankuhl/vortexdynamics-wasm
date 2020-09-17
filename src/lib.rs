mod runge_kutta;
use runge_kutta::{ RungeKuttaSolver, VectorField, Vector };
use ndarray::{ arr1, s, ArrayView1 };
use wasm_bindgen::prelude::*;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

const PI: f64 = 3.14159265358979323;
const STEPSIZE: f64 = 0.001;

fn j_grad_g(x: f64, y: f64, u: f64, v: f64) -> Vector {
    // JGradG(z1, z2) is the product of the 2d symplectic matrix J with the gradient
    // with respect to the first variable of the Green's function G of the negative
    // Dirichlet Laplacian in the unit disk.
    let xnorm = x * x + y * y;
    if u == 0. && v == 0. && 0. < xnorm && xnorm < 1. {
        return arr1(&[-y, x]) / (2. * PI * xnorm)
    } else {
        let unorm = u * u + v * v;
        let prod = x * u + y * v;
        let gx = (unorm - 1.) * (unorm * x + (1. - 2. * v * y) * x + u * (-1. - x * x + y * y));
        let gy = (unorm - 1.) * (unorm * y + (1. - 2. * u * x) * y + v * (-1. + x * x - y * y));
        let denominator = 2. * PI * (unorm + xnorm - 2. * prod) * (1. - 2. * prod + unorm * xnorm);
        arr1(&[gy, -gx]) / denominator
    }
}

fn j_grad_h(x: f64, y: f64) -> Vector {
    // JGradh(z) is the product of the 2d symplectic matrix J with the gradient
    // of the Robin's function of the unit disk.
    if x == 0. && y == 0. {
        return Vector::zeros(2)
    } else {
        let xnorm = x * x + y * y;
        arr1(&[y, -x]) / (PI * (xnorm - 1.))
    }
}

fn hamiltonian_vectorfield(gamma: Vector, z: Vector) -> Vector {
    let dim = gamma.len();
    let x = z.slice(s![0..dim]);
    let y = z.slice(s![dim..]);
    let mut result = Vector::zeros(2 * dim);
    let outer = gamma.iter().zip(x.iter()).zip(y.iter()).enumerate();
    for (j, ((&gamma_j, &x_j), &y_j)) in outer {
        let mut slice = result.slice_mut(s![j..j+2]);
        let gradhj = gamma_j * j_grad_h(x_j, y_j);
        slice += &ArrayView1::<f64>::from(&gradhj);
        let inner = gamma.iter().zip(x.iter()).zip(y.iter()).enumerate();
        for (i, ((&gamma_i, &x_i), &y_i)) in inner {
            if i == j { continue; }
            let gradgij = 2. * gamma_i * j_grad_g(x_j, y_j, x_i, y_i);
            slice += &ArrayView1::<f64>::from(&gradgij);
        }
    }
    result
}

#[wasm_bindgen]
pub struct NVortexProblem {
    solution: RungeKuttaSolver
}

#[wasm_bindgen]
impl NVortexProblem {
    pub fn new(gamma: &[f64], z: &[f64]) -> NVortexProblem {
        let gamma_vector = Vector::from(gamma.to_vec());
        let vector_field = VectorField::Autonomous(Box::new(move |z| hamiltonian_vectorfield(gamma_vector.to_owned(), z)));
        let solution = RungeKuttaSolver::new(vector_field, 0., Vector::from(z.to_vec()), STEPSIZE);
        NVortexProblem { solution: solution }
    }
    pub fn next(&mut self) -> Vec<f64> {
        self.solution.next().unwrap().to_vec()
    }
    pub fn time(&self) -> f64 {
        self.solution.time()
    }
}
