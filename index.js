"use strict"

const tfc = require("@tensorflow/tfjs-core")
const leastSquares = require("tfjs-leastsquares")
const diag = require("tfjs-diag")

const identityLink = {
  link: (mu) => {
    return mu
  },
  inverse: (eta) => {
    return eta
  },
  muEta: (eta) => {
    return tfc.ones(eta.shape)
  }
}

const logitLink = {
  link: (mu) => {
    return tfc.log(tfc.div(mu, tfc.sub(1.0, mu)))
  },
  inverse: (eta) => {
    const expEta = tfc.exp(eta)
    return tfc.div(expEta, tfc.add(1.0, expEta))
  },
  muEta: (eta) => {
    const expEta = tfc.exp(eta)
    return tfc.div(expEta, tfc.square(tfc.add(1.0, expEta)))
  }
}

const logLink = {
  link: (mu) => {
    return tfc.log(mu)
  },
  inverse: (eta) => {
    return tfc.exp(eta)
  },
  muEta: (eta) => {
    return tfc.exp(eta)
  }
}

const getLinkByName = (linkName) => {
  switch (linkName) {
    case "identity":
      return identityLink
    case "logit":
      return logitLink
    case "log":
      return logLink
    default:
      break
  }
}

const binomial = (link) => {
  const linkFun = getLinkByName(link)
  return {
    variance: (mu) => {
      return tfc.mul(mu, tfc.sub(1.0, mu))
    },
    link: linkFun.link,
    linkInverse: linkFun.inverse,
    linkMuEta: linkFun.muEta
  }
}

const gaussian = (link) => {
  const linkFun = getLinkByName(link)
  return {
    variance: (mu) => {
      return tfc.ones(mu.shape)
    },
    link: linkFun.link,
    linkInverse: linkFun.inverse,
    linkMuEta: linkFun.muEta
  }
}

const poisson = (link) => {
  const linkFun = getLinkByName(link)
  return {
    variance: (mu) => {
      return mu
    },
    link: linkFun.link,
    linkInverse: linkFun.inverse,
    linkMuEta: linkFun.muEta
  }
}

// Iteratively Re-Weighted Least Squares
// Page 133, Arnold, T., Kane, M., & Lewis, B. W. (2019). A Computational Approach to Statistical Learning. CRC Press.
const glm = (X, y, family) => {
  return tfc.tidy(() => {
    const maxiter = 25
    let beta = tfc.zeros([X.shape[1], 1])
    for (let i = 0; i < maxiter; i++) {
      const eta = tfc.matMul(X, beta)
      const mu = family.linkInverse(eta)
      const me = family.linkMuEta(eta)
      const W = diag(tfc.sqrt(tfc.divStrict(tfc.square(me), family.variance(mu))).as1D())
      const z = tfc.add(eta, tfc.divStrict(tfc.sub(y, mu), me))
      const newBeta = leastSquares(tfc.matMul(W, X), tfc.matMul(W, z))
      beta = newBeta
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