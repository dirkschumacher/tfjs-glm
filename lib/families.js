"use strict"

const tfc = require("@tensorflow/tfjs-core")
const {identityLink, logitLink, logLink} = require("./links")

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

module.exports = {
  binomial, 
  gaussian, 
  poisson
}
