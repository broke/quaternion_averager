use nalgebra::{
    geometry::Quaternion,
    geometry::UnitQuaternion,
    Matrix4,
    RealField,
    linalg::SymmetricEigen,
};

/// A quaternion averager.
/// Implemented as discussed [here](https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions)
pub struct QuaternionAverager<T: RealField + Copy + PartialEq> {
    matrix: Matrix4<T>,
    weight_sum: T,
}

impl<T: RealField + Copy + PartialEq> QuaternionAverager<T> {
    /// Creates and returns a new quaternion averager
    /// 
    /// # Example
    /// 
    /// ```
    /// use quaternion_averager::QuaternionAverager;
    /// 
    /// let mut qa = QuaternionAverager::<f64>::new();
    /// ```
    pub fn new() -> QuaternionAverager<T> {
        QuaternionAverager {
            matrix: Matrix4::zeros(),
            weight_sum: T::from_f32(0f32).unwrap(),
        }
    }

    /// Add a new unit quaternion with weight 1 to the averager
    /// 
    /// # Example
    /// ```
    /// use quaternion_averager::QuaternionAverager;
    /// use nalgebra::{
    ///     geometry::Quaternion,
    ///     geometry::UnitQuaternion,
    /// };
    /// 
    /// let mut qa = QuaternionAverager::new();
    /// let q1 = Quaternion::new(0.9961947f32, 0.0871557f32, 0f32, 0f32);
    /// let q1 = UnitQuaternion::from_quaternion(q1);
    /// qa.add_quaternion(&q1);
    /// ```
    pub fn add_quaternion(&mut self, quaternion: &UnitQuaternion<T>) {
        let q = quaternion.coords * quaternion.coords.transpose();
        self.matrix += q;
        let w = T::from_f32(1f32).unwrap();
        self.weight_sum += w;
    }

    /// Add a new unit quaternion with custom weight to the averager
    /// 
    /// # Example
    /// ```
    /// use quaternion_averager::QuaternionAverager;
    /// use nalgebra::{
    ///     geometry::Quaternion,
    ///     geometry::UnitQuaternion,
    /// };
    /// 
    /// let mut qa = QuaternionAverager::new();
    /// let q1 = Quaternion::new(0.9961947f32, 0.0871557f32, 0f32, 0f32);
    /// let q1 = UnitQuaternion::from_quaternion(q1);
    /// qa.add_quaternion_weighted(&q1, 0.25f32);
    /// ```
    pub fn add_quaternion_weighted(&mut self, quaternion: &UnitQuaternion<T>, weight: T) {
        let q = quaternion.coords * quaternion.coords.transpose();
        let q = q / weight;
        self.matrix += q;
        self.weight_sum += weight;
    }

    /// Calculates and returns the quaternion average
    /// 
    /// # Example
    /// ```
    /// use quaternion_averager::QuaternionAverager;
    /// use nalgebra::{
    ///     geometry::Quaternion,
    ///     geometry::UnitQuaternion,
    /// };
    /// 
    /// let mut qa = QuaternionAverager::new();
    /// let q1 = Quaternion::new(0.9961947f32, 0.0871557f32, 0f32, 0f32);
    /// let q1 = UnitQuaternion::from_quaternion(q1);
    /// let q2 = Quaternion::new(0.9848078f32, 0.1736482f32, 0f32, 0f32);
    /// let q2 = UnitQuaternion::from_quaternion(q2);
    /// qa.add_quaternion(&q1);
    /// qa.add_quaternion(&q2);
    /// let qavg = qa.calc_average();
    /// println!("The average of {} and {} is {}", q1, q2, qavg);
    /// ```
    pub fn calc_average(&self) -> UnitQuaternion<T> {
        let m = self.matrix / self.weight_sum;
        let decomp = SymmetricEigen::new(m);
        let i = decomp.eigenvalues.imax();
        let q = decomp.eigenvectors.column(i);
        let q = Quaternion::new(q[3], q[0], q[1], q[2]);
        let q = UnitQuaternion::from_quaternion(q);

        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn avg2_32() {
        let mut avg = QuaternionAverager::new();
        let q1 = Quaternion::new(0.9961947f32, 0.0871557f32, 0f32, 0f32);
        let q1 = UnitQuaternion::from_quaternion(q1);
        let q2 = Quaternion::new(0.9848078f32, 0.1736482f32, 0f32, 0f32);
        let q2 = UnitQuaternion::from_quaternion(q2);
        avg.add_quaternion(&q1);
        avg.add_quaternion(&q2);

        let qr = Quaternion::new(0.9914449f32, 0.1305262f32, 0f32, 0f32);
        let qr = UnitQuaternion::from_quaternion(qr);
        let q = avg.calc_average();

        assert_relative_eq!(q, qr, max_relative = 0.00001);
    }

    #[test]
    fn avg2_64() {
        let mut avg = QuaternionAverager::new();
        let q1 = Quaternion::new(0.9961947f64, 0.0871557f64, 0f64, 0f64);
        let q1 = UnitQuaternion::from_quaternion(q1);
        let q2 = Quaternion::new(0.9848078f64, 0.1736482f64, 0f64, 0f64);
        let q2 = UnitQuaternion::from_quaternion(q2);
        avg.add_quaternion(&q1);
        avg.add_quaternion(&q2);

        let qr = Quaternion::new(0.9914449f64, 0.1305262f64, 0f64, 0f64);
        let qr = UnitQuaternion::from_quaternion(qr);
        let q = avg.calc_average();

        assert_relative_eq!(q, qr, max_relative = 0.000001);
    }
}