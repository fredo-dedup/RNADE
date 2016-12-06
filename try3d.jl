using Distributions
# using BenchmarkTools
using VegaLite

# Mixture de Kumaraswamy

Nh = 50  # number of hidden units
Ne = 5   # number of mixture elements
Nd = 50  # variable size to estimate

sigm(x) = 1 ./ (1+exp(-x))

type Pars
  Vm::Vector{Matrix{Float64}}  # for m mixture component
  Vn::Vector{Matrix{Float64}}  # for n mixture component
  Vw::Vector{Matrix{Float64}}  # for weight mixture component
  W::Vector{Matrix{Float64}}
  bm::Vector{Vector{Float64}}  # for m mixture component
  bn::Vector{Vector{Float64}}  # for n mixture component
  bw::Vector{Vector{Float64}}  # for weight mixture component
  c::Vector{Vector{Float64}}
end

function Pars(Nh, Nd, Ne)
    Pars([rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         [rand(Normal(), Nh) for i in 1:Ne]
    )
end

pars = Pars(Nh, Nd, Ne)
sizeof(Pars)
sizeof(pars)
whos(r"pars") # ~400kb

x = vec(draws[1:5,50])

function prob(x::Vector{Float64}, pars::Pars) # x = vec(draws[1:5,50])
  nx = length(x)
  a  = copy(pars.c)
  p  = 1.
  pt = Array(Float64, nx)
  h  = Array(Float64, Nh, nx)
  for i in 1:nx # i = 1
    h[:,i] = sigm(a)
    pxi    = sigm(pars.b[i] + dot(pars.V[:,i], h[:,i]))
    pt[i]  = x[i] > 0.5 ? pxi : (1-pxi)
    p     *= pt[i]
    a     += pars.W[:,i] * x[i]
  end
  p, pt, h
end

prob(draws[:,159], Pars(nH,nD))


function gradp(x::Vector{Float64}, pars::Pars) # x, pars = x₀, pars₀
  nx = length(x)
  pref, pt, h = prob(x, pars)
  δb = zeros(pars.b)
  δh = zeros(h)
  δV = zeros(pars.V)
  δW = zeros(pars.W)
  δlogp = 1.
  δa    = zeros(pars.c)
  for i in nx:-1:1 # i = nx
    δb[i]  = (1 - pt[i]) * (2(x[i] > 0.5) - 1.)

    δV[:,i] = δb[i] * h[:,i]

    δh[:,i] = δb[i] * pars.V[:,i]
    δW[:,i] = δa * x[i]
    δa     += δh[:,i] .* h[:,i] .* (1. - h[:,i])
  end
  (pref, δb, δV, δa, δW)
end



x₀    = draws[:,115]
srand(0)
pars₀ = Pars(nH, nD)
pref, δb, δV, δc, δW = gradp(x₀, pars₀)
logp₀ = log(pref)
logprob(x, pars) = log(prob(x, pars)[1])
logprob(x₀, pars₀) , logp₀

δ = 1e-6
pars = deepcopy(pars₀)
[ δb [ (pars.b=copy(pars₀.b);pars.b[i]+=δ;(logprob(x₀, pars)-logp₀)/δ) for i in 1:nD  ]]
pars = deepcopy(pars₀)
[ δc [ (pars.c=copy(pars₀.c);pars.c[i]+=δ;(logprob(x₀, pars)-logp₀)/δ) for i in 1:nH  ]]
pars = deepcopy(pars₀)
[ vec(δV) [ (pars.V=copy(pars₀.V);pars.V[i]+=δ;(logprob(x₀, pars)-logp₀)/δ) for i in 1:nH*nD  ]]
pars = deepcopy(pars₀)
[ vec(δW) [ (pars.W=copy(pars₀.W);pars.W[i]+=δ;(logprob(x₀, pars)-logp₀)/δ) for i in 1:nH*nD  ]]


function fullgrad(pars::Pars)
  δb = zeros(b)
  δc = zeros(c)
  δV = zeros(V)
  δW = zeros(W)
  logp = 0
  for i in 1:nsamp
    δp, δδb, δδV, δδc, δδW = gradp(draws[:,i], pars)
    logp += log(δp)
    δb += δδb
    δc += δδc
    δV += δδV
    δW += δδW
  end
  (logp, δb, δV, δc, δW)
end

logp, δb, δV, δc, δW = fullgrad(pars₀)

λ = 1e-3
srand(1)
pars = Pars(nH, nD)
for i in 1:50
  logp, δb, δV, δc, δW = fullgrad(pars)
  pars.b += λ*δb
  pars.c += λ*δc
  pars.V += λ*δV
  pars.W += λ*δW
  println(logp)
end


[ prob(draws[:,i], pars)[1] for i in 1:10 ]

log(1/nS)*nS


prob([0.;], pars)[1]
prob([0., 0], pars)[1] # 85%
prob([0., 1], pars)[1] #  8%
prob([1., 1], pars)[1] #  0.7%
prob([1., 0], pars)[1] #  7%
