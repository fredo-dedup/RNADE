############## RNADE param type definition ####################
## with regul


# module RNADE
# end

module RNADE

using Distributions
using BenchmarkTools

abstract NADEDistribution <: Distribution  # distribution adapted to RNADE

import Base: length, rand
import Distributions: logpdf

export NADEDistribution, length, rand, logpdf, dlogpdf!, update!
export Pars, init
export xsample, xdloglik!, xloglik
export score, sgd

# TODO : document
function length(d::NADEDistribution)
  error("length() needs to be defined for $(typeof(d))")
end

# TODO : document
function rand(d::NADEDistribution)
  error("rand() needs to be defined for $(typeof(d))")
end

# TODO : document
function update!(d::NADEDistribution, w::Vector{Float64})
  error("update!() needs to be defined for $(typeof(d))")
end

# TODO : document
function logpdf(d::NADEDistribution, w::Vector{Float64})
  error("logpdf() needs to be defined for $(typeof(d))")
end

# TODO : document
function dlogpdf!(d::NADEDistribution, w::Vector{Float64}, grad::Vector{Float64})
  error("dlogpdf!() needs to be defined for $(typeof(d))")
end


#### RNADE model parameters type definitions  #######################

type Pars{Nh,Nd}
  Vs::Vector{Matrix{Float64}}  # for m mixture component
  bs::Vector{Vector{Float64}}  # for m mixture component
  W::Matrix{Float64}
  c::Vector{Float64}
  dist::NADEDistribution
end

function Pars(d::NADEDistribution, Nh, Nd)
  Ne = length(d)
  Pars{Nh,Nd}([rand(Normal(), Ne, Nh) for i in 1:Nd],
              [rand(Normal(), Ne) for i in 1:Nd],
              rand(Normal(), Nh, Nd),
              rand(Normal(), Nh),
              d)
end

function add!{Nh,Nd}(a::Pars{Nh,Nd}, b::Pars{Nh,Nd})
  for i in 1:Nd
    a.Vs[i] .+= b.Vs[i]
    a.bs[i] .+= b.bs[i]
  end
  a.W .+= b.W
  a.c .+= b.c
  a
end

function scal!{Nh,Nd}(a::Pars{Nh,Nd}, f::Float64)
  for i in 1:Nd
    a.Vs[i] .*= f
    a.bs[i] .*= f
  end
  a.W .*= f
  a.c .*= f
  a
end

import Base.clamp!
function clamp!{Nh,Nd}(a::Pars{Nh,Nd}, low::Float64, up::Float64)
  for i in 1:Nd
    clamp!(a.Vs[i], low, up)
    clamp!(a.bs[i], low, up)
  end
  clamp!(a.W, low, up)
  clamp!(a.c, low, up)
  a
end

import Base.maximum
function maximum{Nh,Nd}(a::Pars{Nh,Nd})
  mx = -Inf
  for i in 1:Nd
    mx = max(mx, maxabs(a.Vs[i]))
    mx = max(mx, maxabs(a.bs[i]))
  end
  mx = max(mx, maxabs(a.W))
  mx = max(mx, maxabs(a.c))
  mx
end

function zeros!{Nh,Nd}(a::Pars{Nh,Nd})
  for i in 1:Nd
    fill!(a.Vs[i], 0.)
    fill!(a.bs[i], 0.)
  end
  fill!(a.W, 0.)
  fill!(a.c, 0.)
  a
end

# xs = rand(20)
# dist = Main.Testing.Bernstein(5)
# pars = Pars(dist,10,50)

####  activation function  ######

# TODO : generalize, make user-settable
sigm(x::Float64) = 1 ./ (1+exp(-x))

const llp_fac = 1.

#### pre-allocating function #######

local a::Vector{Float64}, h::Vector{Vector{Float64}}
local vcpars::Vector{Vector{Float64}}
local δh::Vector{Float64}, δa::Vector{Float64}
local dcpars::Vector{Float64}

function init{Nh,Nd}(pars::Pars{Nh,Nd})
  global  a, h, vcpars, δh, δa, dcpars

  Ne = length(pars.dist)

  a  =     Array(Float64, Nh)
  h  =     [ Array(Float64, Nh) for i in 1:Nd]
  vcpars = [ Array(Float64, Ne) for i in 1:Nd]
  δh =     Array(Float64, Nh)
  δa =     similar(a)
  dcpars = Array(Float64, Ne)
end

#### xloglik!()  function #######

function xloglik{Nh,Nd}(xs::Vector{Float64}, pars::Pars{Nh,Nd})
  # Nh,Nd,N10,50,6
  nx = length(xs)
  a  = copy(pars.c)
  ll  = 0.
  xt = Array(Float64, nx)
  h  = Array(Float64, Nh, nx)
  for i in 1:nx # i = 1
    (xs[i] == -1.) && break
    for j in 1:Nh
      h[j,i] = sigm(a[j])
      # h[:,i] .= max.(0., a)
    end

    update!(pars.dist,
            pars.bs[i] .+ pars.Vs[i] * h[:,i],
            (i==1) ? 0. : xs[i-1] )

    xt[i] = logpdf(pars.dist, xs[i])
    ll += xt[i]
    a .+= pars.W[:,i] * xs[i]
  end

  # penalisation
  llp = 0.
  for i in 1:Nd
    llp += dot(pars.Vs[i],pars.Vs[i])
    # llp += dot(pars.bs[i],pars.bs[i])
  end
  llp += dot(pars.W,pars.W)
  # llp += dot(pars.c,pars.c)

  ll += llp_fac * llp

  ll, xt, h
end

# pars = Pars(10, 50, 7)
# xloglik(rand(30), pars)

#### xdloglik!()  function #######

function xdloglik!{Nh,Nd}(xs::Vector{Float64},
                          pars::Pars{Nh,Nd},
                          dpars::Pars{Nh,Nd})
  # Nh, Nd, Ne = 10, 50, 7
  global δa

  copy!(a, pars.c)
  i = 1
  # for i in 1:nx # i = 1
  while (i <= Nd) && (xs[i] != -1.)
    for j in 1:Nh ; h[i][j] = sigm(a[j]) ; end
    # h[:,i] .= max.(0., a)

    A_mul_B!(vcpars[i], pars.Vs[i], h[i])
    vcpars[i] .+= pars.bs[i]

    a .+= pars.W[:,i] * xs[i]
    i += 1
  end
  nx = i - 1

  zeros!(dpars)

  # δh = Array(Float64, Nh)
  fill!(δa, 0.)

  for i in nx:-1:1 # i = nx-1
    # cpars = reshape(vcpars[i], Ne, 3)
    update!(pars.dist, vcpars[i], (i==1) ? 0. : xs[i-1])
    dlogpdf!(pars.dist, xs[i], dcpars)

    copy!(dpars.bs[i], dcpars)
    A_mul_Bt!(dpars.Vs[i], dcpars, h[i])
    # dpars.Vs[i] = vec(dcpars) * h[i]'

    At_mul_B!(δh, pars.Vs[i], dcpars)

    dpars.W[:,i] = δa * xs[i]
    δa += δh .* h[i] .* (1. - h[i])
    # δa += δh[:,i] .* (h[:,i] .> 0.)
  end
  copy!(dpars.c, δa)

  # penalisation
  for i in 1:Nd
    dpars.Vs[i] += 2 * llp_fac .* pars.Vs[i]
    # dpars.bs[i] += 2 * llp_fac .* pars.bs[i]
  end
  dpars.W += 2 * llp_fac .* pars.W
  # dpars.c += 2 * llp_fac .* pars.c

  dpars
end

# init(pars)
# dpars = Pars(dist, 10, 50)
# xdloglik!(rand(50), pars, dpars)


function xsample{Nh,Nd}(xs::Vector{Float64}, pars::Pars{Nh,Nd})
  nx = length(xs)
  ll = 0.
  xt = Array(Float64, Nd)

  copy!(a, pars.c)

  # known part
  for i in 1:nx # i = 1
    for j in 1:Nh ; h[i][j] = sigm(a[j]) ; end

    A_mul_B!(vcpars[i], pars.Vs[i], h[i])
    vcpars[i] .+= pars.bs[i]
    update!(pars.dist, vcpars[i], (i==1) ? 0. : xs[i-1])
    xt[i] = logpdf(pars.dist, xs[i])
    ll += xt[i]

    a  .+= pars.W[:,i] * xs[i]
  end

  xs2 = zeros(Nd)
  xs2[1:nx] = xs
  # sampled part
  for i in nx+1:Nd # i = nx+1
    for j in 1:Nh ; h[i][j] = sigm(a[j]) ; end
    # h[i] .= sigm.(a)

    A_mul_B!(vcpars[i], pars.Vs[i], h[i])
    vcpars[i] .+= pars.bs[i]
    update!(pars.dist, vcpars[i], xs2[i-1])

    # pick component
    xs2[i] = rand(pars.dist)

    xt[i] = logpdf(pars.dist, xs2[i])
    ll += xt[i]
    a  .+= pars.W[:,i] * xs2[i]
  end

  xs2, xt, ll
end


# end

# pars = Pars(10, 50, 7)
# dpars = deepcopy(pars)
# init(pars)
# # whos(r"pars") # ~400kb
#
# xs = rand(50)
#
# xloglik(xs, pars)
# xdloglik!(xs, pars, dpars)

####### profiling #################

# Profile.clear()
# Base.@profile collect(xdloglik!(rand(50), pars, dpars) for i in 50:5000)
# Profile.print()
#
# @time collect(xdloglik!(rand(50), pars, dpars) for i in 50:500)
# 0.46s  (vs 0.27 best pour Kuma)
# 0.12s

# @benchmark xdloglik!(xs, pars, dpars)
# 136 μs - with immutable PDIST
# 130 μs - with type PDIST


###############  testing ########################

if false
  dist = Main.Testing.Bernstein(10)
  pars = Pars(dist,10,50)
  init(pars)

  dpars = deepcopy(pars); zeros!(dpars)
  xs = rand(50)
  xs[end-5:end] = -1.

  dpars = xdloglik!(xs, pars, dpars)
  v0 = xloglik(xs,pars)[1]

  function dtest(v0, pars, dpars, field, indexes...)
    # field, indexes = :Vm, [2, [1,1]]
    δ = 1e-8

    p0 = foldl((x,idx) -> getindex(x, idx...), getfield(pars, field), indexes)

    pars2 = deepcopy(pars)
    np = foldl((x,idx) -> getindex(x, idx...), getfield(pars2, field), indexes[1:end-1])
    setindex!(np, p0+δ, indexes[end]...)

    ed = (xloglik(xs,pars2)[1]-v0) / δ

    ed0 = foldl((x,idx) -> getindex(x, idx...), getfield(dpars, field), indexes)
    ( ed0, ed )
  end

  ####################

  nd = length(pars.Vs)
  res = Array(Float64, nd, 2)
  for i in 1:nd
    res[i,1], res[i,2] = dtest(v0, pars, dpars, :Vs, i, [1,1])
  end
  res

  for i in 1:nd
    res[i,1], res[i,2] = dtest(v0, pars, dpars, :bs, i, 1)
  end
  res

  for i in 1:nd
    res[i,1], res[i,2] = dtest(v0, pars, dpars, :bs, i, 5)
  end
  res

  dtest(v0, pars, dpars, :Vs, 2, [7,1])
  dtest(v0, pars, dpars, :Vs, 20, [1,10])
  dtest(v0, pars, dpars, :Vs, 50, [5,3])
  dtest(v0, pars, dpars, :Vs, 1, [6,10])

  dtest(v0, pars, dpars, :bs, 2, 1)
  dtest(v0, pars, dpars, :bs, 2, 7)
  dtest(v0, pars, dpars, :bs, 2, 5)
  dtest(v0, pars, dpars, :bs, 2, 2)
  dtest(v0, pars, dpars, :bs, 2, 3)

  dtest(v0, pars, dpars, :bs, 50, 1)
  dtest(v0, pars, dpars, :bs, 50, 7)
  dtest(v0, pars, dpars, :bs, 50, 5)
  dtest(v0, pars, dpars, :bs, 50, 2)
  dtest(v0, pars, dpars, :bs, 50, 6)

  dtest(v0, pars, dpars, :W, [1,1])
  dtest(v0, pars, dpars, :W, [10,1])
  dtest(v0, pars, dpars, :W, [1,10])
  dtest(v0, pars, dpars, :W, [10,30])
  dtest(v0, pars, dpars, :W, [10,15])

  dtest(v0, pars, dpars, :c, 1)
  dtest(v0, pars, dpars, :c, 10)
  dtest(v0, pars, dpars, :c, 5)
  dtest(v0, pars, dpars, :c, 2)
  dtest(v0, pars, dpars, :c, 7)
  dtest(v0, pars, dpars, :c, 9)

end


########### SGD optimization  ##############################

score(pars::Pars, dat) = mean(xloglik(dat[:,j], pars)[1] for j in 1:size(dat,2))

function sgd(pars₀,
             dat,
             datt;
             maxtime=10, maxsteps=1000, chunksize=100,
             kscale=1e-4, cbinterval=100, k0=1e-3)

    α    = 1.      # acceleration parameter
    starttime = time()
    datiter = cycle(1:size(dat,2))
    datstate = start(datiter)

    pars   = deepcopy(pars₀)
    dparsi = deepcopy(pars)
    dpars  = deepcopy(pars)

    for t in 1:maxsteps
        if (maxtime != 0) && (time() - starttime > maxtime)
            break
        end

        zeros!(dpars)
        for i in 1:chunksize
            yi, datstate = next(datiter, datstate)
            add!(dpars, xdloglik!(dat[:,yi], pars, dparsi) )
        end
        dmax = maximum(dpars)
        (1./dmax < α*kscale/chunksize) && print("+")
        scal!(dpars, - min(1./dmax, α*kscale/chunksize))
        # clamp!(scal!(dpars, -α*kscale/chunksize), -1., 1.)
        add!(pars, dpars)
        α = 1. / (1 + t*k0)

        if any(isnan(pars.c))
          break
        end

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            ll  = score(pars,  dat)
            llt = score(pars, datt)
            println("$t : α = $(round(α,3)), train : $(round(ll,1)), test : $(round(llt,1))")
        end
    end

    pars
end


end  # end of module RNADE
