#########################################################################
#  Distribution defined with Berstein polynomials
#   with normalization w = |uw| / Σ |uw|
#   with pdf as defined by distance to bernstein mean
#########################################################################

using Distributions

import .RNADE: length, rand, logpdf, dlogpdf!, update!

type Bernstein4 <: NADEDistribution
  w::Vector{Float64}
  dw::Matrix{Float64}
  binom::Vector{Float64}
  N::Int
  x₀::Float64

  function Bernstein4(n::Int)
    N = n
    binom = Float64[binomial(N-1,i-1) for i in 1:N]
    w = Array(Float64, N)
    dw = Array(Float64, N, N)
    new(w, dw, binom, N, 0.)
  end
end

length(d::Bernstein4) = d.N

function update!(d::Bernstein4, uw::Vector{Float64}, x₀=0.)
  d.x₀ = x₀

  uw2 = abs(uw)
  sw2 = sum(uw2)
  copy!(d.w, uw2 ./ sw2)

  for i in 1:length(uw)
    for j in 1:length(uw)
      d.dw[i,j] = sign(uw[j]) * sign(uw[i]) / sw2 * ( (i==j ? sign(uw[i]) : 0.) - uw[j] / sw2 )
    end
  end

end


# uw = rand(Normal(),5)
#   # uw = [2., -10., 0.]
#   df = Array{Float64,2}(length(uw),length(uw))
#   df2 = Array{Float64,2}(length(uw),length(uw))
#   δ = 1e-8
#   for i in 1:length(uw)
#     uw2 = copy(uw)
#     uw2[i] += δ
#     df[i,:] = (abs(uw2) ./ sum(abs(uw2)) .- abs(uw) ./ sum(abs(uw))) ./ δ
#   end
#
#   uw2 = abs(uw)
#   sw2 = sum(uw2)
#   for i in 1:length(uw)
#     for j in 1:length(uw)
#       df2[i,j] = sign(uw[j]+δ) * sign(uw[i]+δ)/sw2 * ( (i==j ? sign(uw[i]+δ) : 0.) - uw[j] / sw2 )
#     end
#   end
#   hcat(df, df2)
#   norm(df2 - df)
#
# using ReverseDiffSource
# ex = quote
#   uw2 = abs(uw)
#   sw2 = sum(uw2)
#   w = uw2 ./ sw2
#   w[1]
# end
#
# i = 1
# show(rdiff(ex, uw=Vector{Float64}))
# quote
#     _tmp1 = abs(uw)
#     _tmp2 = sum(_tmp1)
#     _tmp3 = size(_tmp1)
#     _tmp4 = _tmp1 ./ _tmp2
#     _tmp5 = zeros(size(_tmp4))
#     _tmp5[1] = 1.0
#     (_tmp4[1],zeros(size(uw)) + sign(uw) .* ((zeros(_tmp3) + _tmp5 ./ _tmp2) + ones(_tmp3) .* (-(sum(_tmp1 .* _tmp5)) / (_tmp2 * _tmp2))))
# end
#
# uw2 = abs(uw)
# sw2 = sum(uw2)
# _tmp3 = size(uw2)
# w = uw2 ./ sw2
# _tmp5 = zeros(size(w))
# _tmp5[i] = 1.0
# (w[i],
#
# sign(uw) ./ sw2 .* (  _tmp5 - ones(_tmp3) .* uw2[i] / sw2  )





############## continuous internal model specs  #############################


### proba
function fp(x::Float64, w::Vector{Float64}, cw::Vector{Float64})
  N = length(w)
  prob = 0.
  for i in 1:N
    dist = (2i-1)/(N+1) - x
    prob += w[i] * exp( - dist*dist*N )
  end
  prob
end

uw = [0., -1, 1, 2, 3, -2]
pd = Bernstein4(length(uw)) ; update!(pd, uw)
#
fp(0.5, pd.w, pd.binom)
# fp(0., pd.w, pd.binom)
# fp(1., pd.w, pd.binom)
#
quadgk(x -> fp(x, pd.w, pd.binom), 0., 1.)
# sum(pd.w[1:3])

### proba + gradient
function dfp!(x::Float64, w::Vector{Float64}, cw::Vector{Float64}, dps::Vector{Float64})
  N = length(w)

  # sum( w[i] * exp( - ( (2i-1)/(N+1) - x ) * N ) for i in 1:N )
  prob = 0.
  for i in 1:N
    dist = (2i-1)/(N+1) - x
    tmp = exp( - dist*dist*N )
    prob += w[i] * tmp
    dps[i] = tmp
  end
  prob
end

# dist = Bernstein4(5)
# dps = zeros(dist.w)
# δ = 1e-8
# w₀ = [1., 1., 1., 1., 1.]
# w₀ ./= sum(w₀)
# v₀ = dfp!(0.2, w₀, dist.binom, dps)
# fp(0.2, w₀, dist.binom)
# [ [ ( fp(0.2, w₀ + δ * eye(5)[i,:], dist.binom) - v₀ ) / δ for i in 1:5 ] dps ]
# # [ fp() ]
#


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
function rand(d::Bernstein4)
  ci = rand(Categorical(d.w))
  ticdf[(d.N-1, ci-1)][rand(1:101)]
end

# uw = [0., -1, 1, 2, 3, -2]
# d = PDist(uw)
# vals = [rand(d, 0.) for i in 1:1000000]
# for x in 0:0.1:1.
#   println(mean(vals .< x), " ", quadgk(x -> fp(x, d.w, d.binom), 0., x)[1] )
# end


####### loglik definitions  #######################################

const pwidth = 1e-2

function logpdf(d::Bernstein4, x::Float64)
  # x = 0.35 ; d = pd ; x₀ = 0.2
  pm = fp(x, d.w, d.binom)
  -log(pm)
end


d = Bernstein4(5)
update!(d, rand(Normal(), length(d)), 0.5)
d

d.dw
# logpdf(pd, 0.9, 0.5)
# uw = rand(Normal(),4)
# pd = PDist(uw)
# quadgk( x -> exp(-logpdf(pd, x, 0.2)), 0., 1.)
# quadgk( x -> exp(-logpdf(pd, x, 0.)), 0., 1.)


# x, x₀ = 0.5, 0.
function dlogpdf!(d::Bernstein4, x::Float64, duw::Vector{Float64})
  pm = dfp!(x, d.w, d.binom, duw)
  copy!(duw, - d.dw * duw ./ pm)
end


#####  testing
if false

function dtest(uw, x, x₀, index)
  # x, x₀, index = 0.5, 0.5, 3
  δ = 1e-8
  w0 = getindex(uw, index)
  d0 = Bernstein4(length(uw))
  update!(d0, uw, x₀)

  uw1 = copy(uw)
  setindex!(uw1, w0+δ, index)
  d1 = Bernstein4(length(uw1))
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
