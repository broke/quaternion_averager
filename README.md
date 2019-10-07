# Description
A small library for quaternion averaging written in rust.

# Usage
``` Rust
use quaternion_averager::QuaternionAverager;
use nalgebra::{
    geometry::Quaternion,
    geometry::UnitQuaternion,
};

let mut qa = QuaternionAverager::new();
let q1 = Quaternion::new(0.9961947f32, 0.0871557f32, 0f32, 0f32);
let q1 = UnitQuaternion::from_quaternion(q1);
let q2 = Quaternion::new(0.9848078f32, 0.1736482f32, 0f32, 0f32);
let q2 = UnitQuaternion::from_quaternion(q2);
qa.add_quaternion(&q1);
qa.add_quaternion_weighted(&q2, 1f32);
let qavg = qa.calc_average();

println!("The average of {} and {} is {}", q1, q2, qavg);
```

# License
MIT or Apache-2.0