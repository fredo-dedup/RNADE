#########################################################################
#  Distribution defined with Berstein polynomials
#  Normalization log(1+exp(uw)), on reduced parameters
#  With dirac density on 0. and 1.
#########################################################################

# w[1:end-2] for weights of Bernstein polynomial
# w[end-1] for weight on 0.
# w[end] for weight on 1.

using Distributions

import .RNADE: length, rand, logpdf, dlogpdf!, update!

type Bernstein7 <: NADEDistribution
  w::Vector{Float64}
  dw::Matrix{Float64}
  binom::Vector{Float64}
  N::Int
  x₀::Float64
end

function Bernstein7(n::Int)
  N = n
  binom = Float64[binomial(N-3,i) for i in 0:N-3]
  w = Array(Float64, N)
  dw = Array(Float64, N, N)
  Bernstein7(w, dw, binom, N, 0.)
end

length(d::Bernstein7) = d.N

function update!(d::Bernstein7, uw::Vector{Float64}, x₀=0.)
  d.x₀ = x₀

  ew = exp(uw)
  ew[1] = 1.
  ewp1 = 1.0 .+ ew
  w2 = log(ewp1)
  sw2 = sum(w2)
  copy!(d.w, w2 ./ sw2)

  copy!(ew, ew ./ ewp1 ./ sw2)  # mem reuse
  d.dw[1,:] = 0.  # zero gradient for uw[1]
  for i in 2:length(uw)
    for j in 1:length(uw)
      d.dw[i,j] = ew[i] * ( Float64(i==j) - d.w[j] )
    end
  end
end

############## continuous internal model specs  #############################

### proba
function fp(x::Float64, w::Vector{Float64}, cw::Vector{Float64})
  N = length(cw)
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
# fp(0.5, pd.w, pd.binom)
# fp(0., pd.w, pd.binom)
# fp(1., pd.w, pd.binom)
# quadgk(x -> fp(x, pd.w, pd.binom), 0., 1.)
# sum(pd.w[1:3])

### proba + gradient
function dfp!(x::Float64, w::Vector{Float64}, cw::Vector{Float64}, dps::Vector{Float64})
  N = length(cw)
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


############## rand() definition  #################

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
for n in 0:20
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

# i, n = 0,0
# ibern.(ticdf[(n,i)],i,n)

# pick a value
function rand(d::Bernstein7)
  ci = rand(Categorical(d.w))
  ci == d.N   && return 1.
  ci == d.N-1 && return 0.
  ticdf[(d.N-3, ci-1)][rand(1:101)]
end

# uw = [0., -1, 1, 2, 1, 2]
# d = Bernstein7(length(uw)) ; update!(d, uw)
# vals = [rand(d) for i in 1:1000000]
# for x in 0:0.1:1.
#   println(mean(vals .<= x), " ", d.w[end]+quadgk(x -> fp(x, d.w, d.binom), 0., x)[1] )
# end
# mean(vals .== 0.)
# mean(vals .== 1.)
# d.w[end-1:end]


####### loglik definitions  #######################################

const pwidth = 1e-2

function logpdf(d::Bernstein7, x::Float64)
  x <=      pwidth && return -log(d.w[end-1] / pwidth)
  x >= 1. - pwidth && return -log(d.w[end]   / pwidth)

  pm = fp(x, d.w, d.binom)
  -log(pm)
end


# d = Bernstein7(5)
# update!(d, rand(Normal(), length(d)), 0.5)
# d
# logpdf(d, 0.01001)
# uw = rand(Normal(),4)
# pd = PDist(uw)
# quadgk( x -> exp(-logpdf(pd, x, 0.2)), 0., 1.)
# quadgk( x -> exp(-logpdf(pd, x, 0.)), 0., 1.)


# x, x₀ = 0.5, 0.
function dlogpdf!(d::Bernstein7, x::Float64, duw::Vector{Float64})
  if x <= pwidth
    pm = d.w[end-1] / pwidth
    fill!(duw, 0.)
    duw[end-1] = 1. / pwidth
  elseif x >= 1. - pwidth
    pm = d.w[end] / pwidth
    fill!(duw, 0.)
    duw[end] = 1. / pwidth
  else
    pm = dfp!(x, d.w, d.binom, duw)
  end
  copy!(duw, - d.dw * duw ./ pm)
end


#####  testing
if false

function dtest(uw, x, x₀, index)
  # x, x₀, index = 0.5, 0.5, 3
  δ = 1e-8
  w0 = getindex(uw, index)
  d0 = Bernstein7(length(uw))
  update!(d0, uw, x₀)

  uw1 = copy(uw)
  setindex!(uw1, w0+δ, index)
  d1 = Bernstein7(length(uw1))
  update!(d1, uw1, x₀)

  ed = (logpdf(d1, x) - logpdf(d0, x)) / δ

  duw = Array(Float64, length(uw))
  dlogpdf!(d0, x, duw)
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

dtest(uw, 0., 0.4, 1)
dtest(uw, 0., 0.4, 2)
dtest(uw, 0., 0.4, 3)
dtest(uw, 0., 0.4, 4)
dtest(uw, 0., 0.4, 5)
dtest(uw, 0., 0.4, 6)

dtest(uw, 1., 1.0, 1)
dtest(uw, 1., 1.0, 2)
dtest(uw, 1., 1.0, 3)
dtest(uw, 1., 1.0, 4)
dtest(uw, 1., 1.0, 5)
dtest(uw, 1., 1.0, 6)

end
