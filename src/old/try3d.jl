using Distributions
# using BenchmarkTools
using VegaLite


include("setup3.jl")

# Mixture de Kumaraswamy

Nh = 50  # number of hidden units
Ne = 5   # number of mixture elements
Nd = 50  # variable size to estimate

sigm(x::Float64) = 1 ./ (1+exp(-x))

type Pars
  Vm::Vector{Matrix{Float64}}  # for m mixture component
  Vn::Vector{Matrix{Float64}}  # for n mixture component
  Vw::Vector{Matrix{Float64}}  # for weight mixture component
  bm::Vector{Vector{Float64}}  # for m mixture component
  bn::Vector{Vector{Float64}}  # for n mixture component
  bw::Vector{Vector{Float64}}  # for weight mixture component
  W::Matrix{Float64}
  c::Vector{Float64}
end

function add!(a::Pars, b::Pars)
  for i in 1:Ne
    a.Vm[i] .+= b.Vm[i]
    a.Vn[i] .+= b.Vn[i]
    a.Vw[i] .+= b.Vw[i]
    a.bm[i] .+= b.bm[i]
    a.bn[i] .+= b.bn[i]
    a.bw[i] .+= b.bw[i]
  end
  a.W .+= b.W
  a.c .+= b.c
  a
end

function scal!(a::Pars, f::Float64)
  for i in 1:Ne
    a.Vm[i] .*= f
    a.Vn[i] .*= f
    a.Vw[i] .*= f
    a.bm[i] .*= f
    a.bn[i] .*= f
    a.bw[i] .*= f
  end
  a.W .*= f
  a.c .*= f
  a
end

function zeros!(a::Pars)
  for i in 1:Ne
    fill!(a.Vm[i], 0.)
    fill!(a.Vn[i], 0.)
    fill!(a.Vw[i], 0.)
    fill!(a.bm[i], 0.)
    fill!(a.bn[i], 0.)
    fill!(a.bw[i], 0.)
  end
  fill!(a.W, 0.)
  fill!(a.c, 0.)
  a
end


function Pars(Nh, Nd, Ne)
    Pars([rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         rand(Normal(), Nh, Nd),
         rand(Normal(), Nh)
    )
end

pars = Pars(Nh, Nd, Ne)
dpars = deepcopy(pars)
whos(r"pars") # ~400kb

xs = rand(20)


function xloglik(xs::Vector{Float64}, pars::Pars)
  nx = length(xs)
  a  = copy(pars.c)
  ll  = 0.
  xt = Array(Float64, nx)
  h  = Array(Float64, Nh, nx)
  cpars = Array(Float64, Ne, 3)
  for i in 1:nx # i = 1
    h[:,i] .= sigm.(a)

    for j in 1:Ne
      m = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      n = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      w = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
      cpars[j,:] = [m,n,w]
    end

    xt[i] = loglik(cpars, xs[i])
    ll += xt[i]
    a  .+= pars.W[:,i] * xs[i]
  end
  ll, xt, h
end

xloglik(xs, pars)


function xdloglik!(xs::Vector{Float64}, pars::Pars, dpars::Pars) # x, pars = x₀, pars₀
  nx = length(xs)
  ll, xt, h = xloglik(xs, pars)

  zeros!(dpars)

  δh = zeros(h)
  δa = zeros(pars.c)

  cpars = Array(Float64, Ne, 3)
  dcpars = similar(cpars)
  for i in nx:-1:1 # i = nx
    for j in 1:Ne
      m = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      n = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      w = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
      cpars[j,:] = [m,n,w]
    end

    dloglik!(cpars, dcpars, xs[i])

    for j in 1:Ne
      dpars.bm[j][i] = dcpars[j,1]
      dpars.bn[j][i] = dcpars[j,2]
      dpars.bw[j][i] = dcpars[j,3]

      dpars.Vm[j][:,i] = dcpars[j,1] .* h[:,i]
      dpars.Vn[j][:,i] = dcpars[j,2] .* h[:,i]
      dpars.Vw[j][:,i] = dcpars[j,3] .* h[:,i]

      δh[:,i] += pars.Vm[j][:,i] .* dcpars[j,1] +
                 pars.Vn[j][:,i] .* dcpars[j,2] +
                 pars.Vw[j][:,i] .* dcpars[j,3]
    end

    dpars.W[:,i] = δa * xs[i]
    δa     += δh[:,i] .* h[:,i] .* (1. - h[:,i])
  end
  copy!(dpars.c, δa)
  dpars
end

xloglik(xs, pars)
xdloglik!(xs, pars, dpars)
dpars.c[1]
dpars.c

δ = 1e-8

###########   c   ok
pars2 = deepcopy(pars)
pars2.c[1]
pars2.c[1] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.c[1]

pars2 = deepcopy(pars)
pars2.c[20]
pars2.c[20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.c[20]

pars2 = deepcopy(pars)
pars2.c[40]
pars2.c[40] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.c[40]

############### W ok
pars2 = deepcopy(pars)
pars2.W[10,10]
pars2.W[10,10] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.W[10,10]


pars2 = deepcopy(pars)
pars2.W[40,10]
pars2.W[40,10] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.W[40,10]

##########  b
# bn OK (convexité élevée !)
pars2 = deepcopy(pars)
pars2.bn[2][20]
pars2.bn[2][20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bn[2][20]

pars2 = deepcopy(pars)
pars2.bn[1][4]
pars2.bn[1][4] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bn[1][4]

pars2 = deepcopy(pars)
pars2.bn[5][35]
pars2.bn[5][35] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bn[5][35]

# bw , OK
pars2 = deepcopy(pars)
pars2.bw[2][30]
pars2.bw[2][30] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bw[2][30]

pars2 = deepcopy(pars)
pars2.bw[3][6]
pars2.bw[3][6] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bw[3][6]

pars2 = deepcopy(pars)
pars2.bw[2][20]
pars2.bw[2][20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bw[2][20]


pars2 = deepcopy(pars)
pars2.bw[2][13]
pars2.bw[2][13] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bw[2][13]

# bm, OK
pars2 = deepcopy(pars)
pars2.bm[5][1]
pars2.bm[5][1] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bm[5][1]

pars2 = deepcopy(pars)
pars2.bm[4][20]
pars2.bm[4][20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bm[4][20]

pars2 = deepcopy(pars)
pars2.bm[4][40]
pars2.bm[4][40] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bm[4][40]


########  W  OK
# Ww, ok
pars2 = deepcopy(pars)
pars2.Vw[2][5,10]
pars2.Vw[2][5,10] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.Vw[2][5,10]

# Wm, ok
pars2 = deepcopy(pars)
pars2.Vw[5][15,10]
pars2.Vw[5][15,10] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.Vw[5][15,10]

# Wn, ok
pars2 = deepcopy(pars)
pars2.Vn[1][20,20]
pars2.Vn[1][20,20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.Vn[1][20,20]


# Wn, ok
pars2 = deepcopy(pars)
pars2.Vn[1][20,40]
pars2.Vn[1][20,40] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.Vn[1][20,40]





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
