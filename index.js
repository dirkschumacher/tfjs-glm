"use strict"

const tfc = require("@tensorflow/tfjs-core")
const leastSquares = require("tfjs-leastsquares")
const diag = require("tfjs-diag")
const {binomial, gaussian, poisson} = require("./lib/families")

// Iteratively Re-Weighted Least Squares
// Page 133, Arnold, T., Kane, M., & Lewis, B. W. (2019). A Computational Approach to Statistical Learning. CRC Press.
// builds the graph for one fitting step
const glmFitStep = (X, y, family, beta) => {
  return tfc.tidy(() => {
    const eta = tfc.matMul(X, beta)
    const mu = family.linkInverse(eta)
    const me = family.linkMuEta(eta)
    const W = diag(tfc.sqrt(tfc.divStrict(tfc.square(me), family.variance(mu))).as1D())
    const z = tfc.add(eta, tfc.divStrict(tfc.sub(y, mu), me))
    return leastSquares(tfc.matMul(W, X), tfc.matMul(W, z))
  })
}

const glm = (X, y, family, maxiter = 25) => {
  return tfc.tidy(() => {
    let beta = tfc.variable(tfc.zeros([X.shape[1], 1]))
    for (let i = 0; i < maxiter; i++) {
      beta.assign(glmFitStep(X, y, family, beta))
    }
    return beta
  })
}

module.exports = {
  glm, 
  binomial, 
  gaussian, 
  poisson
}
