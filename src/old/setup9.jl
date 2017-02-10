################################################################
#
#  setup 9
#  kuma a,b = exp(), + ponctuel 0 , x et 1.
#  activation sigmoide
#
################################################################

using Distributions
# using BenchmarkTools
using VegaLite
using ReverseDiffSource


@time for i in 1:1e6 ; 3.5 ^ 1.123 ; end  # 0.106s
@time for i in 1:1e6 ; exp(log(3.5) * 1.123) ; end # 0.05 plus rapide !!!

############## internal model specs  #############################

# type inference does not work ^ => creation of a power function

power(a::Float64,b::Float64) = exp(log(a)*b)
@deriv_rule power(x::Real, y::Real)  x     y * power(x,y-1.) * ds
@deriv_rule power(x::Real, y::Real)  y     log(x) * power(x,y) * ds

ek3 = quote
    a = exp(m)
    b = exp(n)
    a*b*power(x,a-1.) * power(1. - power(x,a), b-1.)
end

# TODO : optim du code généré

@eval fk3(x::Float64,m::Float64,n::Float64) = $ek3
edk3 = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x, allorders=false)
@eval fdk3(x::Float64,m::Float64,n::Float64) = $edk3
edk3a = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x)
@eval fdk3a(x::Float64,m::Float64,n::Float64) = $edk3a

δ = 1e-8
# pars1 = copy(pars) ; pars1[1,1] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# pars1 = copy(pars) ; pars1[1,2] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# pars1 = copy(pars) ; pars1[1,3] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# dloglik(pars, 0.9)[6,:]

function fcdf3(m::Float64,n::Float64,x::Float64)
  a = exp(m)
  b = exp(n)
  1. - (1 - x^a)^b
end

# fcdf3(-5.,10.,0.5)
# fcdf3.(-2.,2.,[0:0.1:1;])

function ficdf3(m::Float64,n::Float64,p::Float64)
  a = exp(m)
  b = exp(n)
  (1. - (1. - p)^(1/b))^(1/a)
end

# fcdf3(0.,2.,0.5)
# ficdf3(0.,2.,0.361)
# ficdf3(0.,2.,0.5)
ficdf3(-1.,-1.,0.5)

####### loglik definitions  #######################################

function loglik(cpars::Matrix{Float64}, x::Float64, x₀::Float64)
    ne = size(cpars,1)
    ws = exp.(cpars[:,3])
    ws ./= sum(ws)

    pm = 0.
    for i in 1:ne-3
        pm += ws[i] * fk3(x, cpars[i,1], cpars[i,2])
    end

    x00 = clamp(x₀-5e-4, 0.   , 0.999)
    x01 = clamp(x₀+5e-4, 0.001, 1.   )
    x00 < x < x01 && (pm += ws[ne-1]*1000.)
    x>0.999 && (pm += ws[ne]*1000.)
    x<0.001 && (pm += ws[ne-2]*1000.)

    -log(pm)
end
# loglik(pars, 0.5, 0.5)
# exp(-loglik(pars, 0.1))
#
# @benchmark loglik(pars, 0.5) # 28us


# delta = 1e-4
# pars1 = copy(pars) ; pars1[1,1] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# pars1 = copy(pars) ; pars1[1,2] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# pars1 = copy(pars) ; pars1[1,3] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# dloglik(pars, 0.9)[6,:]

x, x₀ = 0.5, 0.3
function dloglik!(cpars::Matrix{Float64}, dcpars::Matrix{Float64},
                  x::Float64, x₀::Float64)
    # p = 0.5
    # x, x₀ = 0.5, 0.5
    ne = size(cpars,1)
    ws0 = exp.(cpars[:,3])
    sws0 = sum(ws0)
    ws = ws0 ./ sws0
    ps = Array(Float64, ne)
    for i in 1:ne-3
      ps[i] = fk3(x, cpars[i,1], cpars[i,2])
    end
    x00 = clamp(x₀-5e-4, 0.   , 0.999)
    x01 = clamp(x₀+5e-4, 0.001, 1.   )
    ps[ne-1] = (x00 < x < x01) * 1000.
    ps[ne]   = (x > 0.999) * 1000.
    ps[ne-2] = (x < 0.001) * 1000.

    pm = dot(ws, ps)

    _tmp5 = - ps ./ pm
    dcpars[:,3] = ws0 ./ sws0 .* ( _tmp5 + ( sum(-ws0 .* _tmp5) / sws0 ) )

    dv0 = -1/pm
    for i in 1:ne-3
        dcpars[i,1], dcpars[i,2] = fdk3(x, cpars[i,1], cpars[i,2])
        dcpars[i,1] *= ws[i] * dv0
        dcpars[i,2] *= ws[i] * dv0
    end
    dcpars[ne-1,1], dcpars[ne-1,2] = 0., 0.
    dcpars[ne  ,1], dcpars[ne  ,2] = 0., 0.
    dcpars[ne-2,1], dcpars[ne-2,2] = 0., 0.

    any(isnan(dcpars)) ? zeros(cpars) : dcpars
end

if false
  function dtest(cpars, x, x₀, indexes)
    # field, indexes = :Vm, [2, [1,1]]
    δ = 1e-8

    p0 = getindex(cpars, indexes...)
    cpars2 = deepcopy(cpars)
    setindex!(cpars2, p0+δ, indexes...)

    ed = (loglik(cpars2, x, x₀) - loglik(cpars, x, x₀)) / δ

    dcpars = zeros(cpars)
    dloglik!(cpars, dcpars, x, x₀)
    ed0 = getindex(dcpars, indexes...)
    ( ed0, ed )
  end

  cpars = rand(5,3)

  dtest(cpars, 0.5, 0.4, [1,1])
  dtest(cpars, 0.5, 0.4, [2,2])

  dtest(cpars, 0.5, 0.4, [5,1])
  dtest(cpars, 0.5, 0.4, [5,2])
  dtest(cpars, 0.5, 0.4, [3,1])
  dtest(cpars, 0.5, 0.4, [4,2])

  dtest(cpars, 0.5, 0.4, [1,3])
  dtest(cpars, 0.5, 0.4, [5,3])
  dtest(cpars, 0.5, 0.4, [5,3])

  dtest(cpars, 0.5, 0.5, [1,1])
  dtest(cpars, 0.5, 0.5, [2,2])

  dtest(cpars, 0.5, 0.5, [1,3])
  dtest(cpars, 0.5, 0.5, [3,3])
  dtest(cpars, 0.5, 0.5, [4,3])
  dtest(cpars, 0.5, 0.5, [5,3])

  dtest(cpars, 0.0001, 0.5, [1,1])
  dtest(cpars, 0.0001, 0.5, [2,2])

  dtest(cpars, 0.0001, 0.5, [1,3])
  dtest(cpars, 0.0001, 0.5, [3,3])
  dtest(cpars, 0.0001, 0.5, [4,3])
  dtest(cpars, 0.0001, 0.5, [5,3])

  dtest(cpars, 0.0001, 0.000, [1,1])
  dtest(cpars, 0.0001, 0.000, [2,2])

  dtest(cpars, 0.0001, 0.000, [1,3])
  dtest(cpars, 0.0001, 0.000, [3,3])
  dtest(cpars, 0.0001, 0.000, [4,3])
  dtest(cpars, 0.0001, 0.000, [5,3])

  dtest(cpars, 0.9995, 0.000, [1,1])
  dtest(cpars, 0.9995, 0.000, [2,2])

  dtest(cpars, 0.9995, 0.000, [1,3])
  dtest(cpars, 0.9995, 0.000, [3,3])
  dtest(cpars, 0.9995, 0.000, [4,3])
  dtest(cpars, 0.9995, 0.000, [5,3])

  dtest(cpars, 0.9995, 1., [1,1])
  dtest(cpars, 0.9995, 1., [2,2])

  dtest(cpars, 0.9995, 1., [1,3])
  dtest(cpars, 0.9995, 1., [3,3])
  dtest(cpars, 0.9995, 1., [4,3])
  dtest(cpars, 0.9995, 1., [5,3])
end

############## RNADE param type definition ####################

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

import Base.clamp!
function clamp!(a::Pars, low::Float64, up::Float64)
  for i in 1:Ne
    clamp!(a.Vm[i], low, up)
    clamp!(a.Vn[i], low, up)
    clamp!(a.Vw[i], low, up)
    clamp!(a.bm[i], low, up)
    clamp!(a.bn[i], low, up)
    clamp!(a.bw[i], low, up)
  end
  clamp!(a.W, low, up)
  clamp!(a.c, low, up)
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



pars = Pars(Nh, Nd, Ne)
dpars = deepcopy(pars)
# whos(r"pars") # ~400kb

xs = rand(20)

############  RNADE defs #######################################

sigm(x::Float64) = 1 ./ (1+exp(-x))

# xs = test_set[:,225]

function xloglik(xs::Vector{Float64}, pars::Pars)
  nx = length(xs)
  a  = copy(pars.c)
  ll  = 0.
  xt = Array(Float64, nx)
  h  = Array(Float64, Nh, nx)
  cpars = Array(Float64, Ne, 3)
  for i in 1:nx # i = 33
    h[:,i] .= sigm.(a)
    # h[:,i] .= max.(0., a)

    for j in 1:Ne
      cpars[j,1] = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      cpars[j,2] = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      cpars[j,3] = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
    end

    xt[i] = loglik(cpars, xs[i], (i==1) ? xs[1] : xs[i-1])
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
      cpars[j,1] = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      cpars[j,2] = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      cpars[j,3] = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
    end

    dloglik!(cpars, dcpars, xs[i], (i==1) ? xs[1] : xs[i-1])

    for j in 1:Ne
      local m,n,w
      dm,dn,dw = dcpars[j,1], dcpars[j,2], dcpars[j,3]
      dpars.bm[j][i] = dm
      dpars.bn[j][i] = dn
      dpars.bw[j][i] = dw

      dpars.Vm[j][:,i] = dm .* h[:,i]
      dpars.Vn[j][:,i] = dn .* h[:,i]
      dpars.Vw[j][:,i] = dw .* h[:,i]

      δh[:,i] += pars.Vm[j][:,i] .* dm +
                 pars.Vn[j][:,i] .* dn +
                 pars.Vw[j][:,i] .* dw
    end

    dpars.W[:,i] = δa * xs[i]
    δa     += δh[:,i] .* h[:,i] .* (1. - h[:,i])
    # δa += δh[:,i] .* (h[:,i] .> 0.)
  end
  copy!(dpars.c, δa)
  dpars
end

xloglik(xs, pars)
xdloglik!(xs, pars, dpars)
dpars.c[1]
dpars.c

# profiling #################

# Profile.clear()
# Base.@profile collect(xdloglik!(dat[:,i], pars, dpars) for i in 50:500)
# Profile.print()
#
@time collect(xdloglik!(dat[:,i], pars, dpars) for i in 50:500)
# 3.83 s
# 3.14 s
# 3.00 s
# 2.86 s
# 1.81 s
# 0.71 s

###############  testing ########################

if false
pars = scal!(Pars(Nh, Nd, Ne), 0.1)
dpars = deepcopy(pars)
xs = rand(50)

xs = train_set[:,400]

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


dtest(v0, pars, dpars, :Vm, 2, [1,1])
dtest(v0, pars, dpars, :Vm, 2, [10,1])
dtest(v0, pars, dpars, :Vm, 2, [1,10])
dtest(v0, pars, dpars, :Vm, 2, [10,30])
dtest(v0, pars, dpars, :Vm, 2, [10,15])
dtest(v0, pars, dpars, :Vm, 2, [10,50])

dtest(v0, pars, dpars, :Vm, 5, [1,1])
dtest(v0, pars, dpars, :Vm, 5, [10,1])
dtest(v0, pars, dpars, :Vm, 5, [1,10])
dtest(v0, pars, dpars, :Vm, 5, [10,30])
dtest(v0, pars, dpars, :Vm, 5, [10,15])
dtest(v0, pars, dpars, :Vm, 5, [10,50])

dtest(v0, pars, dpars, :Vn, 2, [1,1])
dtest(v0, pars, dpars, :Vn, 2, [10,1])
dtest(v0, pars, dpars, :Vn, 2, [1,10])
dtest(v0, pars, dpars, :Vn, 2, [10,30])
dtest(v0, pars, dpars, :Vn, 2, [10,15])
dtest(v0, pars, dpars, :Vn, 2, [10,50])

dtest(v0, pars, dpars, :Vn, 5, [1,1])
dtest(v0, pars, dpars, :Vn, 5, [10,1])
dtest(v0, pars, dpars, :Vn, 5, [1,10])
dtest(v0, pars, dpars, :Vn, 5, [10,30])
dtest(v0, pars, dpars, :Vn, 5, [10,15])

dtest(v0, pars, dpars, :Vw, 2, [1,1])
dtest(v0, pars, dpars, :Vw, 2, [10,1])
dtest(v0, pars, dpars, :Vw, 2, [1,10])
dtest(v0, pars, dpars, :Vw, 2, [10,30])
dtest(v0, pars, dpars, :Vw, 2, [10,15])

dtest(v0, pars, dpars, :Vw, 5, [1,1])
dtest(v0, pars, dpars, :Vw, 5, [10,1])
dtest(v0, pars, dpars, :Vw, 5, [1,10])
dtest(v0, pars, dpars, :Vw, 5, [10,30])
dtest(v0, pars, dpars, :Vw, 5, [10,15])



dtest(v0, pars, dpars, :bm, 2, 1)
dtest(v0, pars, dpars, :bm, 2, 10)
dtest(v0, pars, dpars, :bm, 2, 15)
dtest(v0, pars, dpars, :bm, 2, 20)
dtest(v0, pars, dpars, :bm, 2, 50)

dtest(v0, pars, dpars, :bm, 5, 1)
dtest(v0, pars, dpars, :bm, 5, 10)
dtest(v0, pars, dpars, :bm, 5, 15)
dtest(v0, pars, dpars, :bm, 5, 20)
dtest(v0, pars, dpars, :bm, 5, 45)
dtest(v0, pars, dpars, :bm, 5, 50)

dtest(v0, pars, dpars, :bn, 2, 1)
dtest(v0, pars, dpars, :bn, 2, 10)
dtest(v0, pars, dpars, :bn, 2, 15)
dtest(v0, pars, dpars, :bn, 2, 20)
dtest(v0, pars, dpars, :bn, 2, 45)

dtest(v0, pars, dpars, :bn, 5, 1)
dtest(v0, pars, dpars, :bn, 5, 10)
dtest(v0, pars, dpars, :bn, 5, 15)
dtest(v0, pars, dpars, :bn, 5, 20)
dtest(v0, pars, dpars, :bn, 5, 45)

dtest(v0, pars, dpars, :bw, 2, 1)
dtest(v0, pars, dpars, :bw, 2, 10)
dtest(v0, pars, dpars, :bw, 2, 15)
dtest(v0, pars, dpars, :bw, 2, 20)
dtest(v0, pars, dpars, :bw, 2, 45)

dtest(v0, pars, dpars, :bw, 5, 1)
dtest(v0, pars, dpars, :bw, 5, 10)
dtest(v0, pars, dpars, :bw, 5, 15)
dtest(v0, pars, dpars, :bw, 5, 20)
dtest(v0, pars, dpars, :bw, 5, 45)


dtest(v0, pars, dpars, :W, [1,1])
dtest(v0, pars, dpars, :W, [10,1])
dtest(v0, pars, dpars, :W, [1,10])
dtest(v0, pars, dpars, :W, [10,30])
dtest(v0, pars, dpars, :W, [10,15])

dtest(v0, pars, dpars, :c, 1)
dtest(v0, pars, dpars, :c, 10)
dtest(v0, pars, dpars, :c, 5)

end

##########################################################################
## make a sample following the partial values given
##########################################################################

function xsample(xs::Vector{Float64}, pars::Pars)
  nx = length(xs)
  a  = copy(pars.c)
  ll = 0.
  xt = Array(Float64, Nd)
  h  = Array(Float64, Nh, Nd)
  cpars = Array(Float64, Ne, 3)

  # known part
  for i in 1:nx # i = 1
    h[:,i] .= sigm.(a)

    for j in 1:Ne
      cpars[j,1] = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      cpars[j,2] = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      cpars[j,3] = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
    end

    prevxt = (i==1) ? xs[1] : xs[i-1]
    xt[i] = loglik(cpars, xs[i], prevxt)
    ll += xt[i]
    a .+= pars.W[:,i] * xs[i]
  end

  xs2 = zeros(Nd)
  xs2[1:nx] = xs
  # sampled part
  for i in nx+1:Nd # i = nx+1
    h[:,i] .= sigm.(a)

    for j in 1:Ne
      cpars[j,1] = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      cpars[j,2] = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      cpars[j,3] = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
    end

    # pick component
    wn = exp.(cpars[:,3])
    wn ./= sum(wn)
    ci = rand(Categorical(wn))

    # pick x value
    if ci == Ne   # on 1.
        xs2[i] = 0.9999
    elseif ci == Ne-1 # centered on x(t-1)
        xs2[i] = xs2[i-1]
    elseif ci == Ne-2  # on 0.
        xs2[i] = 0.0001
    else
        xs2[i] = ficdf3(cpars[ci,1], cpars[ci,2], rand())
    end

    xt[i] = loglik(cpars, xs2[i], xs2[i-1])
    ll += xt[i]
    a  .+= pars.W[:,i] * xs2[i]
  end

  xs2, xt, ll
end
