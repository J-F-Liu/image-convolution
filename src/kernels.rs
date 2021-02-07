use crate::Real;

/// Square shaped convolution kernel
pub struct Kernel {
    pub data: Vec<Real>,
    pub size: u32,
}

impl std::ops::Index<(u32, u32)> for Kernel {
    type Output = Real;

    #[inline]
    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        let idx = (y * self.size + x) as usize;
        &self.data[idx]
    }
}

/// compute gaussian kernel
pub fn gaussian(sigma: Real) -> Kernel {
    /*
      The size of the kernel is selected to guarantee that the first discarded
      term is at least 10^prec times smaller than the central value. For that,
      the half size of the kernel must be larger than x, with
        e^(-x^2/2sigma^2) = 1/10^prec
      Then,
        x = sigma * sqrt( 2 * prec * ln(10) )
    */
    let prec = 3.0;
    let radius = (sigma * (2.0 * prec * (10.0 as Real).ln()).sqrt()).ceil() as i32;
    let size = 1 + 2 * radius; /* kernel size */
    let mut data = Vec::with_capacity((size * size) as usize);
    for y in -radius..=radius {
        for x in -radius..=radius {
            let dist2 = x.pow(2) + y.pow(2);
            // proximate a circle region
            let value = if dist2 <= radius * radius {
                (-0.5 * (dist2 as Real) / sigma.powi(2)).exp()
            } else {
                0.0
            };
            data.push(value);
        }
    }

    //normalization
    let sum: Real = data.iter().sum();
    if sum > 0.0 {
        for v in data.iter_mut() {
            *v /= sum;
        }
    }

    Kernel {
        data,
        size: size as u32,
    }
}

pub fn roberts_operator() -> (Kernel, Kernel) {
    let kx = Kernel {
        data: vec![1.0, 0.0, 0.0, -1.0],
        size: 2,
    };
    let ky = Kernel {
        data: vec![0.0, 1.0, -1.0, 0.0],
        size: 2,
    };
    (kx, ky)
}

pub fn desolneux_operator() -> (Kernel, Kernel) {
    let kx = Kernel {
        data: vec![-0.5, 0.5, -0.5, 0.5],
        size: 2,
    };
    let ky = Kernel {
        data: vec![-0.5, -0.5, 0.5, 0.5],
        size: 2,
    };
    (kx, ky)
}

pub fn sobel_operator() -> (Kernel, Kernel) {
    let kx = Kernel {
        data: vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
        size: 3,
    };
    let ky = Kernel {
        data: vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
        size: 3,
    };
    (kx, ky)
}

pub fn freichen_operator() -> (Kernel, Kernel) {
    let sqrt_2 = (2.0 as Real).sqrt();
    let kx = Kernel {
        data: vec![-1.0, 0.0, 1.0, -sqrt_2, 0.0, sqrt_2, -1.0, 0.0, 1.0],
        size: 3,
    };
    let ky = Kernel {
        data: vec![-1.0, -sqrt_2, -1.0, 0.0, 0.0, 0.0, 1.0, sqrt_2, 1.0],
        size: 3,
    };
    (kx, ky)
}
