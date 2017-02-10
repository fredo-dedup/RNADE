################################################################
#
#  using Bernstein polynomials
#  with with no dirac on 0, x0 and 1
#
################################################################

module DModel

using Distributions
# using VegaLite
# using ReverseDiffSource

import Distributions: pdf, logpdf, rand
export logpdf, dlogpdf!, PDist, rand, updatePDist!

# calculates normalized probability weights & jacobian
function _updatePDIST!(uw::Vector{Float64}, w::Vector{Float64}, dw::Matrix{Float64})
  ew = exp(uw)
  ewp1 = 1.0 .+ ew
  w2 = log(ewp1)
  sw2 = sum(w2)
  copy!(w, w2 ./ sw2)

  copy!(ew, ew ./ ewp1 ./ sw2)  # mem reuse
  for i in 1:length(uw)
    for j in 1:length(uw)
      dw[i,j] = ew[i] * ( Float64(i==j) - w[j] )
    end
  end

  # copy!(dw, ew ./ ewp1 ./ sw2 .* ( eye(length(uw)) .- w' ))
end

# distrib type
type PDist
  w::Vector{Float64}
  dw::Matrix{Float64}
  binom::Vector{Float64}
  N::Int

  function PDist(uw::Vector{Float64})
    N = length(uw)
    binom = Float64[binomial(N-1,i-1) for i in 1:N]
    w = Array(Float64, N)
    dw = Array(Float64, N, N)
    _updatePDIST!(uw, w, dw)

    new(w, dw, binom, N)
  end
end


#  weights update
updatePDist!(d::PDist, uw::Vector{Float64}) = _updatePDIST!(uw, d.w, d.dw)

# pd = PDist([-1., 0., 1., 5.])
# pd.dw * ones(4)
#
# function pdf(d::PDist, x)
#   d.w[2]
# end
#
# δ = 1e-8
#
# uw = [-1., 0., 1., 5.]
# uw2 = [-1.+δ, 0., 1., 5.]
# (pdf(PDist(uw2), 4.) - pdf(PDist(uw), 4.)) / δ
#
# uw = [-1., 0., 1., 5.]
# uw2 = [-1., 0.+δ, 1., 5.]
# (pdf(PDist(uw2), 4.) - pdf(PDist(uw), 4.)) / δ
#
# uw = [-1., 0., 1., 5.]
# uw2 = [-1., 0., 1.+δ, 5.]
# (pdf(PDist(uw2), 4.) - pdf(PDist(uw), 4.)) / δ
#
# uw = [-1., 0., 1., 5.]
# uw2 = [-1., 0., 1., 5.+δ]
# (pdf(PDist(uw2), 4.) - pdf(PDist(uw), 4.)) / δ
#
#
# PDist(uw).dw



############## continuous internal model specs  #############################

### Bernstein polynomials Distribution

### proba
function fp(x::Float64, w::Vector{Float64}, cw::Vector{Float64})
  N = length(w)
  ps = zeros(N)
  vx = 1.
  for i in 1:N
    ps[i] = vx * w[i] * cw[i]
    vx = vx * x
  end

  vx = 1.
  for i in N:-1:1
    ps[i] = ps[i] * vx
    vx = vx * (1. - x)
  end

  sum(ps) * N
end

# uw = [0., -1, 1, 2, 3, -2]
# pd = PDist(uw)
#
# fp(0.5, pd.w, pd.binom)
# fp(0., pd.w, pd.binom)
# fp(1., pd.w, pd.binom)
#
# quadgk(x -> fp(x, pd.w, pd.binom), 0., 1.)
# sum(pd.w[1:3])

### proba + gradient
function dfp!(x::Float64, w::Vector{Float64}, cw::Vector{Float64}, dps::Vector{Float64})
  N = length(w)
  ps = zeros(N)
  vx = 1.
  for i in 1:N
    ps[i] = vx * w[i] * cw[i]
    dps[i] = vx * cw[i]
    vx = vx * x
  end

  vx = 1.
  for i in N:-1:1
    ps[i] *= vx
    dps[i] *= vx * N
    vx = vx * (1. - x)
  end

  sum(ps) * N
end

# dps = zeros(length(pd.w))
# dfp!(0.5, pd.w, pd.binom, dps)
# dps
#
# pd2 = deepcopy(pd); pd2.w[1] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[2] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[3] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[4] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[5] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[6] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ


### gradient only
function dfp2!(x::Float64, w::Vector{Float64}, cw::Vector{Float64}, dps::Vector{Float64})
  N = length(w)
  vx = 1.
  for i in 1:N
    dps[i] = vx * cw[i]
    vx = vx * x
  end

  vx = 1.
  for i in N:-1:1
    dps[i] *= vx * N
    vx = vx * (1. - x)
  end
end



############## full model (continuous part + steps at 0, 1. and x0  #################

#  cdf tables for quick inverse cdf calculations
bern(x::Float64, i::Int, n::Int) = (n+1) * binomial(n,i) * x^i * (1. - x)^(n-i)
ibern(x::Float64, i::Int, n::Int) = sum( bern(x, j, n+1) for j in i+1:n+1 ) / (n+2)
# x1, i, n = 0.8, 0, 1
# quadgk(x -> bern(x,i,n), 0, x1)
# ibern(x1,i,n)

function findx(target::Float64, x₀::Float64, i::Int, n::Int; tol=1e-5)
  # target, x₀, i, n = 0.01, 0.01, 2, 3
  # tol = 1e-5
  maxiter = 100
  k = 0
  x, y = x₀, ibern(x₀,i,n)
  xmin, xmax = 0., 1.
  while abs(y - target) > tol && k < maxiter
    dist = y - target
    if dist > 0.
      xmax = min(xmax, x)
    else
      xmin = max(xmin, x)
    end
    δ = bern(x,i,n)
    x -= δ == 0. ? sign(dist)/100 : dist / bern(x,i,n)
    (x >= xmax || x <= xmin) && (x = (xmax + xmin) / 2)
    # x = clamp(x, xmin, xmax)
    y = ibern(x,i,n)
    k += 1
  end
  k == maxiter && error("no convergence for $target - $x₀ - $i - $n")
  x
end

ticdf = Dict{Tuple{Int,Int}, Array{Float64,1}}()
for n in 0:10
  for i in 0:n
    x₀ = 0.5
    xs = Array(Float64, 101)
    for (pos, y) in enumerate(0:0.01:1)
      x₀ = findx(y, x₀, i, n)
      xs[pos] = x₀
    end
    ticdf[(n,i)] = xs
  end
end

i, n = 0,0
ibern.(ticdf[(n,i)],i,n)

# pick a value
function rand(d::PDist, x₀::Float64)
  ci = rand(Categorical(d.w))
  ticdf[(d.N-1, ci-1)][rand(1:101)]
end

# uw = [0., -1, 1, 2, 3, -2]
# d = PDist(uw)
# vals = [rand(d, 0.) for i in 1:1000000]
# for x in 0:0.1:1.
#   println(mean(vals .< x), " ", quadgk(x -> fp(x, d.w, d.binom), 0., x)[1] )
# end


####### loglik definitions w/  #######################################

const pwidth = 1e-2

function logpdf(d::PDist, x::Float64, x₀::Float64)
  # x = 0.35 ; d = pd ; x₀ = 0.2
  pm = fp(x, d.w, d.binom)

  -log(pm)
end

# logpdf(pd, 0.9, 0.5)

# uw = rand(Normal(),4)
# pd = PDist(uw)
# quadgk( x -> exp(-logpdf(pd, x, 0.2)), 0., 1.)
# quadgk( x -> exp(-logpdf(pd, x, 0.)), 0., 1.)


# x, x₀ = 0.5, 0.
function dlogpdf!(d::PDist, x::Float64, x₀::Float64, duw::Vector{Float64})
  pm = dfp!(x, d.w, d.binom, duw)
  copy!(duw, - d.dw * duw ./ pm)
end


#####  testing
if false

δ = 1e-8
function dtest(uw, x, x₀, index)
  # x, x₀, index = 0.5, 0.5, 3
  w0 = getindex(uw, index)
  d0 = PDist(uw)
  uw1 = copy(uw)
  setindex!(uw1, w0+δ, index)
  d1 = PDist(uw1)

  ed = (logpdf(d1, x, x₀) - logpdf(d0, x, x₀)) / δ

  duw = Array(Float64, length(uw))
  dlogpdf!(d0, x, x₀, duw)
  ed0 = getindex(duw, index)
  ( ed0, ed )
end

uw = rand(Normal(), 6)

dtest(uw, 0.5, 0.4, 1)
dtest(uw, 0.5, 0.4, 2)
dtest(uw, 0.5, 0.4, 3)
dtest(uw, 0.5, 0.4, 4)
dtest(uw, 0.5, 0.4, 5)
dtest(uw, 0.5, 0.4, 6)

dtest(uw, 0.5, 0.499, 1)
dtest(uw, 0.5, 0.499, 2)
dtest(uw, 0.5, 0.499, 3)
dtest(uw, 0.5, 0.499, 4)
dtest(uw, 0.5, 0.499, 5)
dtest(uw, 0.5, 0.499, 6)

dtest(uw, 0., 0.4, 1)
dtest(uw, 0., 0.4, 2)
dtest(uw, 0., 0.4, 3)
dtest(uw, 0., 0.4, 4)
dtest(uw, 0., 0.4, 5)
dtest(uw, 0., 0.4, 6)

dtest(uw, 0., 0.0, 1)
dtest(uw, 0., 0.0, 2)
dtest(uw, 0., 0.0, 3)
dtest(uw, 0., 0.0, 4)
dtest(uw, 0., 0.0, 5)
dtest(uw, 0., 0.0, 6)

dtest(uw, 1., 0.4, 1)
dtest(uw, 1., 0.4, 2)
dtest(uw, 1., 0.4, 3)
dtest(uw, 1., 0.4, 4)
dtest(uw, 1., 0.4, 5)
dtest(uw, 1., 0.4, 6)

dtest(uw, 1., 1.0, 1)
dtest(uw, 1., 1.0, 2)
dtest(uw, 1., 1.0, 3)
dtest(uw, 1., 1.0, 4)
dtest(uw, 1., 1.0, 5)
dtest(uw, 1., 1.0, 6)

end

end # module
