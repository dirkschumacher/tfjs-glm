"use strict"

const test = require("tape")
const tfc = require("@tensorflow/tfjs-core")
const {glm, gaussian, binomial} = require(".")
const mtcars = require("mtcars")
const round = require("lodash.round")

test("fit mtcars guassian identity", (t) => {
  const mpg = mtcars.map((x) => x.mpg)
  const n = mpg.length
  const m = 2
  const hp = mtcars.map((x) => x.hp)
  const cyl = mtcars.map((x) => x.cyl)
  const response = tfc.tensor2d(mpg, [n, 1])
  
  const designMatrix = tfc.tensor2d(
    [hp, cyl],
    [m, n]
  ).transpose()

  // fit the model
  const coefficents = glm(designMatrix, response, gaussian("identity"))

  // computed with R 3.4.2
  // fitted coefficents
  const expectedHp = -0.107465705415024
  const expectedCyl = 5.403644695759401

  const arrayEqual = (a, b) => {
    for(let i = 0; i < n; i++) {
      t.equal(a[i], b[i])
    }
  }
  const rm = (x) => round(x, 4)
  arrayEqual(coefficents.arraySync().map(rm), [expectedHp, expectedCyl].map(rm))

  t.end()
})

test("fit mtcars binomial logit", (t) => {
  const mpg = mtcars.map((x) => x.mpg < 20 ? 1 : 0)
  const n = mpg.length
  const m = 2
  const hp = mtcars.map((x) => x.hp)
  const cyl = mtcars.map((x) => x.cyl)
  const response = tfc.tensor2d(mpg, [n, 1])
  
  const designMatrix = tfc.tensor2d(
    [cyl, hp],
    [m, n]
  ).transpose()

  // fit the model
  const coefficents = glm(designMatrix, response, binomial("logit"))

  const expectedHp = 0.03582136
  const expectedCyl = -0.68511922
  
  const arrayEqual = (a, b) => {
    for(let i = 0; i < n; i++) {
      t.equal(a[i], b[i])
    }
  }
  const rm = (x) => round(x, 4)
  arrayEqual(coefficents.arraySync().map(rm), [expectedCyl, expectedHp].map(rm))

  t.end()
})


//test("fit mtcars poisson log", (t) => {
//  const hp = mtcars.map((x) => x.hp)
//  const n = hp.length
//  const m = 2
//  const mpg = mtcars.map((x) => x.mpg)
//  const cyl = mtcars.map((x) => x.cyl)
//  const response = tfc.tensor2d(hp, [n, 1])
//  
//  const designMatrix = tfc.tensor2d(
//    [cyl, mpg],
//    [m, n]
//  ).transpose()
//  
//  // fit the model
//  const coefficents = glm(designMatrix, response, poisson("log"))
//
//  const expectedMpg = 0.07801086
//  const expectedCyl = 0.52441693
//  
//  const arrayEqual = (a, b) => {
//    for(let i = 0; i < n; i++) {
//      t.equal(a[i], b[i])
//    }
//  }
//  const rm = (x) => round(x, 4)
//  arrayEqual(coefficents.arraySync().map(rm), [expectedCyl, expectedMpg].map(rm))
//
//  t.end()
//})