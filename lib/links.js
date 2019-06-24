"use strict"

const tfc = require("@tensorflow/tfjs-core")

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
    return tfc.tidy(() => {
      return tfc.log(tfc.div(mu, tfc.sub(1.0, mu)))
    })
  },
  inverse: (eta) => {
    return tfc.tidy(() => {
      const expEta = tfc.exp(eta)
      return tfc.div(expEta, tfc.add(1.0, expEta))
    })
  },
  muEta: (eta) => {
    return tfc.tidy(() => {
      const expEta = tfc.exp(eta)
      return tfc.div(expEta, tfc.square(tfc.add(1.0, expEta)))
    })
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

module.exports = {identityLink, logitLink, logLink}
