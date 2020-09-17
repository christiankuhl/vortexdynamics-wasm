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
const C: [f64; 5] = [0.25, 0.375, 12./13., 1., 0.5];

pub enum RungeKuttaSolver { 
    Autonomous(RungeKuttaSolverAutonomous),
    NonAutonomous(RungeKuttaSolverNonAutonomous)
}

pub struct RungeKuttaSolverAutonomous {
    f: Box<dyn Fn(Vector) -> Vector>,
    h: f64,
    tn: f64, 
    xn: Vector,
    dim: usize
}

pub struct RungeKuttaSolverNonAutonomous {
    f: Box<dyn Fn(f64, Vector) -> Vector>,
    h: f64,
    tn: f64, 
    xn: Vector,
    dim: usize
}

impl RungeKuttaSolver {
    pub fn new(vector_field: VectorField, t0: f64, x0: Vector, step_size: f64) -> RungeKuttaSolver {
        let dim = x0.len();
        match vector_field {
            VectorField::Autonomous(vector_field) => {
                RungeKuttaSolver::Autonomous(RungeKuttaSolverAutonomous { f: vector_field, tn: t0, xn: x0, h: step_size, dim: dim })
            }
            VectorField::NonAutonomous(vector_field) => {
                RungeKuttaSolver::NonAutonomous(RungeKuttaSolverNonAutonomous { f: vector_field, tn: t0, xn: x0, h: step_size, dim: dim })
            }
        }
    }
    pub fn time(&self) -> f64 {
        match self {
            Self::Autonomous(solver) => solver.tn,
            Self::NonAutonomous(solver) => solver.tn,
        }
    }        
}

impl RungeKuttaSolverAutonomous {
    fn runge_kutta_step(&mut self) -> Vector {
        let mut x_tmp = Vector::zeros(self.dim);
        x_tmp.assign(&self.xn);
        let k0 = (*self.f)(x_tmp);
        let mut k = [k0, Vector::zeros(self.dim), Vector::zeros(self.dim), Vector::zeros(self.dim), 
                     Vector::zeros(self.dim), Vector::zeros(self.dim), Vector::zeros(self.dim)];
        for (j, _) in C.iter().enumerate() {
            let delta_x: Vector = A[j][0 .. j+1].iter()
                                    .zip(&k[0 .. j+1])
                                    .map(|(aji, ki)| self.h * aji * ki)
                                    .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
            let mut x_tmp = Vector::zeros(self.dim);
            x_tmp.assign(&self.xn);
            k[j + 1] = (*self.f)(x_tmp + delta_x);
        }
        self.xn += &B.iter().zip(&k)
                    .map(|(&bi, ki)| bi * ki)
                    .fold(Vector::zeros(self.dim), 
                        |sum, elem| sum + self.h * elem);
        self.tn += self.h;
        let mut result = Vector::zeros(self.dim);
        result.assign(&self.xn);
        result
    }
}

impl RungeKuttaSolverNonAutonomous {
    fn runge_kutta_step(&mut self) -> Vector {
        let mut x_tmp = Vector::zeros(self.dim);
        x_tmp.assign(&self.xn);
        let k0 = (*self.f)(self.tn, x_tmp);
        let mut k = [k0, Vector::zeros(self.dim), Vector::zeros(self.dim), Vector::zeros(self.dim), 
                     Vector::zeros(self.dim), Vector::zeros(self.dim), Vector::zeros(self.dim)];
        for (j, c) in C.iter().enumerate() {
            let delta_x: Vector = A[j][0 .. j+1].iter().zip(&k[0 .. j+1])
                                    .map(|(aji, ki)| self.h * aji * ki)
                                    .fold(Vector::zeros(self.dim), |sum, elem| sum + elem);
            let mut x_tmp = Vector::zeros(self.dim);
            x_tmp.assign(&self.xn);
            k[j + 1] = (*self.f)(self.tn + c * self.h, delta_x + x_tmp);
        }
        self.xn += &B.iter().zip(&k)
                    .map(|(&bi, ki)| bi * ki)
                    .fold(Vector::zeros(self.dim), 
                        |sum, elem| sum + self.h * elem);
        self.tn += self.h;
        let mut result = Vector::zeros(self.dim);
        result.assign(&self.xn);
        result
    }
}

impl Iterator for RungeKuttaSolver {
    type Item = Vector;
    fn next(&mut self) -> Option<Vector> {
        let result;
        match self {
            Self::Autonomous(solver) => {
                result = solver.runge_kutta_step();
            }
            Self::NonAutonomous(solver) => {
                result = solver.runge_kutta_step();
            }
        };
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

    #[test]
    fn rotation_vectorfield_is_periodic() {
        let rotation = VectorField::Autonomous(Box::new(rotate));
        let mut solution = RungeKuttaSolver::new(rotation, 0., arr1(&[1., 0.]), 0.001);
        let mut x = Some(Vector::zeros(2));
        while solution.time() < 2. * 3.141592653 {
            x = solution.next();
            let time = solution.time();
            println!("{}: {:?}", time, x);
        }
        let delta = x.unwrap() - arr1(&[1., 0.]);
        let norm2 = delta.dot(&delta);
        println!("{}", norm2);
        let error_small = norm2 < 1e-6;
        assert!(error_small);
    }

    #[test]
    fn attractor_is_attractive() {
        let attractor = VectorField::Autonomous(Box::new(|v: Vector| -v));
        let mut solution = RungeKuttaSolver::new(attractor, -1., arr1(&[1., 1.]), 0.01);
        let mut x = Some(Vector::zeros(2));
        while solution.time() < 2. * 3.141592653 {
            x = solution.next();
            let time = solution.time();
            println!("{}: {:?}", time, x);
        }
        let x = x.unwrap();
        let norm2 = x.dot(&x);
        println!("{}", norm2);
        let error_small = norm2 < 1e-6;
        assert!(error_small);
    }
}